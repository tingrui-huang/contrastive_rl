"""Stage 3A evaluation: does the support score encode CONDITIONAL support?

Loads the trained behavior-vs-target discriminators and evaluates them on the
FINAL TEST episodes (the Stage-1 validation episodes, never used for model
selection) against one cached diagnostic bank, so every model -- D_B, D_A,
action-only and context-only -- sees exactly the same states, commanded goals,
real actions, query goals and CRL actions.

Reminder carried through every output: the score is a RELATIVE support /
discrepancy score, p_behavior/(p_behavior + p_target) under an artificial 50/50
class prior. It is NOT P(A=a|S=s,G=g), NOT a propensity, and NOT a causal
mixture weight.

Diagnostics beyond raw classification, in order of importance:

  A  correct-context vs shuffled-context real action
  B  full model vs the action-only baseline
  C  cross-context behavior action (same context, another episode's action)
  D  local perturbation probe around a real behavior action
  +  boundary-only shortcut control on THIS exact real-vs-CRL action bank
  +  A-vs-B score agreement (does g_cmd change the model here?)
  +  stochastic CRL sample vs deterministic mode sensitivity

No environment is constructed and no rollout is generated. BehaviorFlow is not
imported or used anywhere in this file.

Run:

  python -m propensity.eval_discriminator \
      --model B=artifacts/support_discriminator/D_state_cmdgoal_action \
      --model A=artifacts/support_discriminator/D_state_action \
      --model action=artifacts/support_discriminator/D_action \
      --model context=artifacts/support_discriminator/D_context
"""
import argparse
import json
import os
import pickle
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(_HERE) not in sys.path:
  sys.path.insert(0, os.path.dirname(_HERE))

import jax
import jax.numpy as jnp
import numpy as np

from propensity import crl_policy_adapter as crl_adapter
from propensity import discriminator as disc_mod
from propensity.dataset import BehaviorDataset
from propensity.train_discriminator import make_splits


def _boundary_helpers():
  """Lazy import of the Stage-2.5 boundary helpers.

  Imported inside the function so this module's import graph stays free of
  propensity.flow (audit_boundary_shortcut pulls it in for the archived
  BehaviorFlow comparison). The helpers used here are pure numpy."""
  from propensity.audit_boundary_shortcut import (boundary_features,
                                                  fit_logistic, grouped_split,
                                                  roc_auc)
  return boundary_features, fit_logistic, grouped_split, roc_auc

PERTURB_DELTAS = (0.0, 0.02, 0.05, 0.10, 0.20, 0.50)


def _spearman(x, y):
  rx = np.argsort(np.argsort(np.asarray(x, float)))
  ry = np.argsort(np.argsort(np.asarray(y, float)))
  return float(np.corrcoef(rx, ry)[0, 1])


def _avg_precision(scores, labels):
  from sklearn.metrics import average_precision_score
  return float(average_precision_score(labels, scores))


def _bce(logits, labels):
  z = np.asarray(logits, float)
  y = np.asarray(labels, float)
  return float(np.mean(np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))))


def _quantiles(x):
  q = np.percentile(np.asarray(x, float), [5, 25, 50, 75, 95])
  return {'mean': float(np.mean(x)), 'median': float(q[2]),
          'q05': float(q[0]), 'q25': float(q[1]), 'q75': float(q[3]),
          'q95': float(q[4])}


def load_model(run_dir):
  with open(os.path.join(run_dir, 'config.json')) as f:
    meta = json.load(f)
  with open(os.path.join(run_dir, 'model.pkl'), 'rb') as f:
    blob = pickle.load(f)
  cfg = disc_mod.DiscriminatorConfig(
      input_dim=meta['architecture']['input_dim'],
      hidden_sizes=tuple(meta['architecture']['hidden_sizes']))
  net = disc_mod.make_discriminator(cfg)
  std = disc_mod.Standardizer(mean=blob['standardizer']['mean'],
                              std=blob['standardizer']['std'])
  params = jax.tree_util.tree_map(jnp.asarray, blob['params'])
  fn = jax.jit(lambda x: net.apply(params, std.apply(x)))
  return meta, fn


def build_parser():
  p = argparse.ArgumentParser(
      description='Stage 3A evaluation of relative support discriminators.')
  p.add_argument('--model', action='append', required=True,
                 help='label=run_dir (repeatable)')
  p.add_argument('--n-test', type=int, default=8192)
  p.add_argument('--seed', type=int, default=0)
  p.add_argument('--out', default='artifacts/support_discriminator/eval.json')
  return p


def main(argv=None):
  args = build_parser().parse_args(argv)
  boundary_features, fit_logistic, grouped_split, roc_auc = _boundary_helpers()
  models = {}
  for spec in args.model:
    label, _, path = spec.partition('=')
    models[label] = load_model(path)
  ref = next(iter(models.values()))[0]           # shared provenance

  ds = BehaviorDataset(ref['dataset_path'], val_frac=ref['val_frac'],
                       seed=ref['stage1_split_seed'], state_mode='obs',
                       split_level='episode')
  if ds.fingerprint['sha256'] != ref['dataset_sha256']:
    print('ABORT: dataset sha256 mismatch vs the training record.')
    return 1
  S, A = ds._obs_dim, ds.action_dim                            # noqa: SLF001
  gidx = ref['cmdgoal_indices']
  rows_by_split, eps = make_splits(ds, ref['dev_episodes']
                                   if 'dev_episodes' in ref
                                   else len(ref['episodes']['dev']),
                                   ref['dev_split_seed'])
  test_rows = rows_by_split['test']
  lengths = ds._lengths                                        # noqa: SLF001
  ep_of_row = ds._episode_of_row                               # noqa: SLF001
  starts = np.concatenate([[0], np.cumsum(lengths - 1)])
  t_of_row = np.arange(ds.n_transitions) - starts[ep_of_row]

  # ---- one cached test bank, shared by every model ------------------------ #
  rng = np.random.default_rng(args.seed)
  n = min(args.n_test, len(test_rows))
  sel = test_rows[rng.choice(len(test_rows), size=n, replace=False)]
  raw = ds._state                                              # noqa: SLF001
  s = np.asarray(raw[sel][:, :S])
  g = np.asarray(raw[sel][:, S:][:, gidx])
  a_real = np.asarray(ds._action[sel])                         # noqa: SLF001
  groups = ep_of_row[sel]

  goal_src = crl_adapter.load_goal_source(ref['dataset_path'], S)
  j_future = crl_adapter.sample_future_goal_index(
      lengths, ep_of_row[sel], t_of_row[sel],
      np.random.default_rng(args.seed + 11))
  g_query = np.asarray(goal_src[ep_of_row[sel], j_future])
  _, crl_apply, crl_info = crl_adapter.load_frozen_crl_actor(
      ref['crl']['checkpoint'], S, ds._goal_dim, A)            # noqa: SLF001
  eps_bank = np.random.default_rng(args.seed + 12).standard_normal((n, A))
  a_crl, a_crl_mode = crl_adapter.crl_actions(crl_apply, s, g_query, eps_bank)

  # derangement across the bank, used by diagnostics A and C
  perm = (np.arange(n) + 1 + rng.integers(0, n - 1)) % n
  assert np.all(perm != np.arange(n))
  cross_episode = float(np.mean(groups[perm] != groups))

  print('=' * 80)
  print('propensity.eval_discriminator -- Stage 3A relative support score')
  print('=' * 80)
  print(f'dataset       {os.path.basename(ref["dataset_path"])}  '
        f'sha {ref["dataset_sha256"][:16]}...')
  print(f'CRL actor     {os.path.basename(os.path.dirname(ref["crl"]["checkpoint"]))}'
        f'/{os.path.basename(ref["crl"]["checkpoint"])} @ step '
        f'{crl_info["step"]} (FROZEN, no rollouts)')
  print(f'TEST bank     {n} contexts from {len(np.unique(groups))} held-out '
        f'episodes (Stage-1 val; never used for selection)')
  print(f'g_query       {ref["query_goal_semantics"]["probability_rule"]}, '
        f'discount {ref["query_goal_semantics"]["discount"]}')
  print(f'negatives     one fixed STOCHASTIC pi sample per context; '
        f'mode reported as a sensitivity check')
  print(f'shuffle       derangement; {cross_episode:.4f} of pairs cross '
        f'episodes')
  print('SCORE IS A RELATIVE SUPPORT/DISCREPANCY SCORE -- NOT A PROPENSITY')

  report = {'n_test_contexts': int(n),
            'test_episodes': [int(x) for x in np.unique(groups)],
            'cross_episode_shuffle_fraction': cross_episode,
            'score_semantics': ref['score_semantics'],
            'NOT_a_propensity': True,
            'behaviorflow_used': False,
            'environment_rollouts_used': False,
            'models': {}}

  def score(fn, meta, st, gg, aa):
    x = disc_mod.assemble_inputs(meta['input_spec'], st, gg, aa)
    return np.asarray(fn(x))

  # ======================================================================== #
  # Primary classification metrics
  # ======================================================================== #
  print()
  print('PRIMARY CLASSIFICATION (held-out TEST episodes, real vs CRL)')
  hdr = (f'{"model":<10} {"input spec":<22} {"ROC-AUC":>8} {"PR-AUC":>8} '
         f'{"acc":>7} {"balAcc":>7} {"BCE":>8}')
  print(hdr)
  print('-' * len(hdr))
  logit_cache = {}
  for label, (meta, fn) in models.items():
    lp = score(fn, meta, s, g, a_real)
    ln = score(fn, meta, s, g, a_crl)
    logit_cache[label] = (lp, ln)
    sc = np.concatenate([lp, ln])
    y = np.concatenate([np.ones(n, int), np.zeros(n, int)])
    pred = (sc >= 0).astype(int)
    m = {'roc_auc': roc_auc(sc, y), 'pr_auc': _avg_precision(sc, y),
         'accuracy': float((pred == y).mean()),
         'balanced_accuracy': float(0.5 * ((pred[:n] == 1).mean()
                                           + (pred[n:] == 0).mean())),
         'bce': _bce(sc, y),
         'score_real': _quantiles(lp), 'score_crl': _quantiles(ln),
         'input_spec': meta['input_spec'],
         'dev_auc_at_selection': meta['best_dev_auc'],
         'selected_step': meta['best_step']}
    report['models'][label] = m
    print(f'{label:<10} {meta["input_spec"]:<22} {m["roc_auc"]:>8.4f} '
          f'{m["pr_auc"]:>8.4f} {m["accuracy"]:>7.4f} '
          f'{m["balanced_accuracy"]:>7.4f} {m["bce"]:>8.4f}')

  print()
  print('SCORE DISTRIBUTIONS (raw logits)')
  print(f'{"model":<10} {"class":<6} {"mean":>8} {"median":>8} {"q05":>8} '
        f'{"q25":>8} {"q75":>8} {"q95":>8}')
  for label in models:
    for cls, k in (('real', 'score_real'), ('CRL', 'score_crl')):
      q = report['models'][label][k]
      print(f'{label:<10} {cls:<6} {q["mean"]:>8.3f} {q["median"]:>8.3f} '
            f'{q["q05"]:>8.3f} {q["q25"]:>8.3f} {q["q75"]:>8.3f} '
            f'{q["q95"]:>8.3f}')

  # ======================================================================== #
  # Mandatory boundary-only shortcut control on THIS bank
  # ======================================================================== #
  xp, names = boundary_features(a_real)
  xn, _ = boundary_features(a_crl)
  xb = np.concatenate([xp, xn], axis=0)
  yb = np.concatenate([np.ones(n, int), np.zeros(n, int)])
  gb = np.concatenate([groups, groups])
  tr_b, te_b, ng_tr, ng_te = grouped_split(gb, 0.3, 0)

  class _A:
    test_frac, split_seed = 0.3, 0
  bnd = fit_logistic(xb[tr_b], yb[tr_b], xb[te_b], yb[te_b], names, 0)
  print()
  print('BOUNDARY-ONLY SHORTCUT CONTROL (real vs CRL, this exact bank)')
  print(f'  boundary-only logistic AUC : {bnd["test_auc"]:.4f}  '
        f'(previous audit: 0.519; chance = 0.5)')
  print(f'  balanced accuracy          : {bnd["test_balanced_accuracy"]:.4f}')
  report['boundary_shortcut_control'] = {
      'auc': bnd['test_auc'],
      'balanced_accuracy': bnd['test_balanced_accuracy'],
      'previous_audit_auc': 0.519,
      'note': 'real behavior actions carry no boundary source signature; this '
              'is why generated positives were dropped'}

  # ======================================================================== #
  # Conditional-support diagnostics
  # ======================================================================== #
  print()
  print('CONDITIONAL-SUPPORT DIAGNOSTICS')
  for label, (meta, fn) in models.items():
    parts = disc_mod.INPUT_SPECS[meta['input_spec']]
    if 'action' not in parts:
      continue
    sc_correct = logit_cache[label][0]
    # A: same real action, WRONG context
    sc_ctx_shuf = score(fn, meta, s[perm], g[perm], a_real)
    # C: same context, another episode's real action
    sc_act_cross = score(fn, meta, s, g, a_real[perm])
    d = {
        'A_correct_context_mean': float(sc_correct.mean()),
        'A_shuffled_context_mean': float(sc_ctx_shuf.mean()),
        'A_paired_difference_mean': float((sc_correct - sc_ctx_shuf).mean()),
        'A_fraction_correct_gt_shuffled': float((sc_correct > sc_ctx_shuf).mean()),
        'C_correct_action_mean': float(sc_correct.mean()),
        'C_cross_context_action_mean': float(sc_act_cross.mean()),
        'C_paired_difference_mean': float((sc_correct - sc_act_cross).mean()),
        'C_fraction_correct_gt_cross': float((sc_correct > sc_act_cross).mean()),
    }
    # D: local perturbation probe, fixed noise directions
    pr = np.random.default_rng(args.seed + 21)
    direction = pr.standard_normal((n, A))
    direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True),
                            1e-8)
    curve = []
    for delta in PERTURB_DELTAS:
      a_d = np.clip(a_real + delta * direction, -1.0, 1.0)
      curve.append(float(score(fn, meta, s, g, a_d).mean()))
    d['D_perturbation_deltas'] = list(PERTURB_DELTAS)
    d['D_perturbation_mean_score'] = curve
    report['models'][label].update(d)
    print(f'  [{label}] A correct {d["A_correct_context_mean"]:+.4f} vs '
          f'shuffled-context {d["A_shuffled_context_mean"]:+.4f} '
          f'(diff {d["A_paired_difference_mean"]:+.4f}, win '
          f'{d["A_fraction_correct_gt_shuffled"]:.4f})')
    print(f'  [{label}] C correct {d["C_correct_action_mean"]:+.4f} vs '
          f'cross-context action {d["C_cross_context_action_mean"]:+.4f} '
          f'(diff {d["C_paired_difference_mean"]:+.4f}, win '
          f'{d["C_fraction_correct_gt_cross"]:.4f})')
    print(f'  [{label}] D perturbation ' +
          '  '.join(f'{dd:.2f}:{cc:+.4f}'
                    for dd, cc in zip(PERTURB_DELTAS, curve)))

  # ======================================================================== #
  # A vs B agreement on TARGET actions
  # ======================================================================== #
  if 'A' in models and 'B' in models:
    sa = logit_cache['A'][1]
    sb = logit_cache['B'][1]
    lo_a, lo_b = np.percentile(sa, 10), np.percentile(sb, 10)
    hi_a, hi_b = np.percentile(sa, 90), np.percentile(sb, 90)
    low_a, low_b = sa <= lo_a, sb <= lo_b
    high_a, high_b = sa >= hi_a, sb >= hi_b
    ab = {
        'pearson': float(np.corrcoef(sa, sb)[0, 1]),
        'spearman': _spearman(sa, sb),
        'mean_abs_score_difference': float(np.abs(sa - sb).mean()),
        'mean_abs_diff_relative_to_score_std':
            float(np.abs(sa - sb).mean() / max(sa.std(), 1e-8)),
        'decision_disagreement_at_zero': float(((sa >= 0) != (sb >= 0)).mean()),
        'lowest_decile_jaccard':
            float((low_a & low_b).sum() / max((low_a | low_b).sum(), 1)),
        'lowest_decile_disagreement':
            float(1 - (low_a & low_b).sum() / max(low_a.sum(), 1)),
        'highest_decile_jaccard':
            float((high_a & high_b).sum() / max((high_a | high_b).sum(), 1)),
        'highest_decile_disagreement':
            float(1 - (high_a & high_b).sum() / max(high_a.sum(), 1)),
    }
    report['A_vs_B'] = ab
    print()
    print('A vs B AGREEMENT ON TARGET (CRL) ACTIONS')
    print(f'  pearson {ab["pearson"]:.4f} | spearman {ab["spearman"]:.4f} | '
          f'mean|diff| {ab["mean_abs_score_difference"]:.4f} '
          f'({ab["mean_abs_diff_relative_to_score_std"]:.3f} of score sd)')
    print(f'  decision disagreement at 0 : '
          f'{ab["decision_disagreement_at_zero"]:.4f}')
    print(f'  lowest-decile  disagreement: '
          f'{ab["lowest_decile_disagreement"]:.4f} '
          f'(jaccard {ab["lowest_decile_jaccard"]:.4f})')
    print(f'  highest-decile disagreement: '
          f'{ab["highest_decile_disagreement"]:.4f} '
          f'(jaccard {ab["highest_decile_jaccard"]:.4f})')

  # ======================================================================== #
  # Stochastic sample vs deterministic mode
  # ======================================================================== #
  print()
  print('CRL NEGATIVE SENSITIVITY: stochastic sample vs deterministic mode')
  sens = {}
  for label, (meta, fn) in models.items():
    lp = logit_cache[label][0]
    ln_mode = score(fn, meta, s, g, a_crl_mode)
    sc = np.concatenate([lp, ln_mode])
    y = np.concatenate([np.ones(n, int), np.zeros(n, int)])
    sens[label] = {'roc_auc_vs_mode': roc_auc(sc, y),
                   'roc_auc_vs_sample': report['models'][label]['roc_auc'],
                   'score_crl_mode': _quantiles(ln_mode)}
    print(f'  {label:<10} AUC vs sample {sens[label]["roc_auc_vs_sample"]:.4f} '
          f'| AUC vs mode {sens[label]["roc_auc_vs_mode"]:.4f}')
  report['crl_mode_sensitivity'] = sens

  os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
  with open(args.out, 'w') as f:
    json.dump(report, f, indent=2)
  print()
  print('REMINDER: sigmoid(logit) is the posterior of an ARTIFICIAL 50/50 '
        'behavior-vs-target\nclassification problem. It is NOT P(A=a|S=s,G=g) '
        'and NOT a causal branch weight.')
  print(f'report -> {os.path.abspath(args.out)}')
  return 0


if __name__ == '__main__':
  sys.exit(main())
