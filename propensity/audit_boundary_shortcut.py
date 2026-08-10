"""Stage 2.5: boundary-shortcut audit (NO discriminator is trained here).

Question: could a future discriminator D(s, g, a) separate BehaviorFlow actions
from CRL target-policy actions using ONLY boundary/saturation artifacts -- i.e.
learn "coordinate exactly at +/-1 => flow sample" instead of learning genuine
behavioral support/overlap?

The audit fits deliberately weak, boundary-ONLY classifiers. They never see the
state, the goal, a context or episode id, a timestep, or which checkpoint an
action came from. The only inputs are functions of the action's distance to the
action box. The point is to measure how far a stupid boundary detector can get.

Three comparisons, all on the SAME held-out contexts:

  A  Flow vs CRL   -- approximates the shortcut available at Stage 3
  B  Real vs CRL   -- CONTROL: does saturation genuinely differ between the
                      behavior policy and the CRL learner, with no flow in the
                      loop? If A ~= B, most of A is a real policy difference.
  C  Flow vs Real  -- measures the artificial signature the generative model
                      introduces on top of the behavior distribution.

Everything is offline: held-out validation transitions are read from the frozen
npz and pushed through frozen networks. No environment is constructed, no
rollout is generated, and nothing is written back to the dataset. BehaviorFlow
and the CRL actor are both loaded read-only and are never modified or retrained.

Run:

  python -m propensity.audit_boundary_shortcut \
      --flow-config artifacts/propensity_flow/rockfall_v2_p30_h800_s0/config.json \
      --flow-ckpt 50k=artifacts/propensity_flow/rockfall_v2_p30_h800_s0_cont150k/checkpoint_50000.pkl \
      --flow-ckpt 150k=artifacts/propensity_flow/rockfall_v2_p30_h800_s0_cont150k/checkpoint_150000.pkl \
      --crl-ckpt naive_rockfall_v2_p30_h800_resetfix_s0_300k/final.pkl
"""
import argparse
import csv
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

BOUNDARY_TOL = 1e-6
MAIN_N = 10                     # CFQL-aligned sampler; unchanged by this audit

#: CRL actor config -- from scripts/verify_offline_d4rl.build_offline_cfg, the
#: config the rockfall runs were trained with. Verified against the checkpoint's
#: parameter shapes at load time (58 -> 1024 -> 1024 -> 8).
CRL_HIDDEN = (1024, 1024)
CRL_REPR_DIM = 16
CRL_TWIN_Q = True
CRL_LAYER_NORM = False


# --------------------------------------------------------------------------- #
# Boundary-only features. No state, goal, index, timestep or provenance.
# --------------------------------------------------------------------------- #
def boundary_features(a, tol=BOUNDARY_TOL):
  """[N, A] actions -> ([N, F] features, feature names).

  Every feature is a function of |a| alone and its distance to the box edge."""
  a = np.asarray(a, dtype=np.float64)
  n, d = a.shape
  absa = np.abs(a)
  exact = (np.abs(absa - 1.0) <= tol).astype(np.float64)        # [N, d]
  dist = 1.0 - absa                                             # [N, d]
  near99 = (absa >= 0.99).astype(np.float64)
  near95 = (absa >= 0.95).astype(np.float64)

  feats, names = [], []
  feats.append(exact.max(axis=1, keepdims=True))                # F1
  names.append('any_exact_boundary')
  feats.append(exact.sum(axis=1, keepdims=True))                # F2
  names.append('count_exact_boundary')
  feats.append(exact)                                           # F3
  names += [f'exact_dim{j}' for j in range(d)]
  feats.append(dist)                                            # F4
  names += [f'dist_dim{j}' for j in range(d)]
  feats.append(dist.min(axis=1, keepdims=True))                 # F5
  names.append('min_dist_to_boundary')
  feats.append(near99)
  names += [f'ge099_dim{j}' for j in range(d)]
  feats.append(near95)
  names += [f'ge095_dim{j}' for j in range(d)]
  return np.concatenate(feats, axis=1), names


def population_stats(a, tol=BOUNDARY_TOL):
  a = np.asarray(a)
  absa = np.abs(a)
  exact = np.abs(absa - 1.0) <= tol
  return {
      'p_any_exact_boundary': float(exact.any(axis=1).mean()),
      'mean_count_exact_boundary': float(exact.sum(axis=1).mean()),
      'p_exact_per_dim': exact.mean(axis=0).tolist(),
      'p_ge099_per_dim': (absa >= 0.99).mean(axis=0).tolist(),
      'p_ge095_per_dim': (absa >= 0.95).mean(axis=0).tolist(),
      'p_exact_overall': float(exact.mean()),
      'p_ge099_overall': float((absa >= 0.99).mean()),
      'p_ge095_overall': float((absa >= 0.95).mean()),
      'mean_dist_to_boundary': float((1.0 - absa).mean()),
      'mean_min_dist_to_boundary': float((1.0 - absa).min(axis=1).mean()),
      'mean_per_dim': a.mean(axis=0).tolist(),
      'std_per_dim': a.std(axis=0).tolist(),
  }


def roc_auc(scores, labels):
  """Rank-based ROC-AUC (ties averaged). No sklearn needed for the scalars."""
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


def grouped_split(groups, test_frac, seed):
  """Split by GROUP (episode) so no context -- and no temporally adjacent
  context from the same episode -- appears on both sides."""
  uniq = np.unique(groups)
  rng = np.random.default_rng(seed)
  perm = rng.permutation(uniq)
  n_test = max(1, int(round(test_frac * len(uniq))))
  test_groups = set(perm[:n_test].tolist())
  is_test = np.array([g in test_groups for g in groups])
  return ~is_test, is_test, len(uniq) - n_test, n_test


def fit_logistic(x_tr, y_tr, x_te, y_te, names, seed=0):
  """Low-capacity boundary-only classifier. Standardization uses TRAIN stats."""
  from sklearn.linear_model import LogisticRegression
  mu, sd = x_tr.mean(axis=0), x_tr.std(axis=0)
  sd = np.where(sd < 1e-12, 1.0, sd)          # constant features -> no scaling
  xtr, xte = (x_tr - mu) / sd, (x_te - mu) / sd
  clf = LogisticRegression(max_iter=2000, random_state=seed)
  clf.fit(xtr, y_tr)
  p = clf.predict_proba(xte)[:, 1]
  pred = (p >= 0.5).astype(int)
  auc = roc_auc(p, y_te)
  acc = float((pred == y_te).mean())
  pos, neg = y_te == 1, y_te == 0
  bal = float(0.5 * ((pred[pos] == 1).mean() + (pred[neg] == 0).mean()))
  coefs = sorted(zip(names, clf.coef_[0].tolist()),
                 key=lambda kv: -abs(kv[1]))
  return {'test_auc': auc, 'test_accuracy': acc, 'test_balanced_accuracy': bal,
          'n_train': int(len(y_tr)), 'n_test': int(len(y_te)),
          'coefficients': [{'feature': k, 'weight': v} for k, v in coefs],
          'top_features': [k for k, _ in coefs[:8]]}


def compare(name, pos, neg, groups, args):
  """One classification comparison. Classes are 1:1 by construction."""
  xp, names = boundary_features(pos)
  xn, _ = boundary_features(neg)
  x = np.concatenate([xp, xn], axis=0)
  y = np.concatenate([np.ones(len(xp), int), np.zeros(len(xn), int)])
  g = np.concatenate([groups, groups])        # same contexts -> same groups
  tr, te, n_gtr, n_gte = grouped_split(g, args.test_frac, args.split_seed)

  idx = {n: i for i, n in enumerate(names)}
  scalars = {
      'any_boundary': roc_auc(x[te, idx['any_exact_boundary']], y[te]),
      'count_boundary': roc_auc(x[te, idx['count_exact_boundary']], y[te]),
      # lower min-distance => closer to the boundary => flip the sign so a
      # higher score always means "more positive-class-like".
      'min_dist_to_boundary': roc_auc(-x[te, idx['min_dist_to_boundary']],
                                      y[te]),
  }
  logi = fit_logistic(x[tr], y[tr], x[te], y[te], names, args.split_seed)
  return {'comparison': name,
          'n_pos': int(len(xp)), 'n_neg': int(len(xn)),
          'pos_p_any_boundary': float(np.mean(
              np.abs(np.abs(pos) - 1.0).min(axis=1) <= BOUNDARY_TOL)),
          'neg_p_any_boundary': float(np.mean(
              np.abs(np.abs(neg) - 1.0).min(axis=1) <= BOUNDARY_TOL)),
          'train_episodes': n_gtr, 'test_episodes': n_gte,
          'scalar_auc': scalars, 'logistic': logi}


FEATURE_FAMILIES = {
    'all_boundary_features': lambda nm: True,
    'exact_boundary_only': lambda nm: 'exact' in nm,
    'near_boundary_only': lambda nm: nm.startswith('ge0'),
    'distance_only': lambda nm: nm.startswith('dist_') or nm == 'min_dist_to_boundary',
}


def feature_ablation(pos, neg, groups, args):
  """Which FAMILY of boundary features carries the shortcut?

  Separates two very different diagnoses:
    * exact-boundary features carry it  -> a hard-CLIPPING artifact, which a
      bounded-flow reparameterization would plausibly remove;
    * near-boundary features carry it   -> the learned DENSITY is wrong in the
      shell just inside the box, which removing the clip would NOT fix.
  """
  xp, names = boundary_features(pos)
  xn, _ = boundary_features(neg)
  x_all = np.concatenate([xp, xn], axis=0)
  y = np.concatenate([np.ones(len(xp), int), np.zeros(len(xn), int)])
  g = np.concatenate([groups, groups])
  tr, te, _, _ = grouped_split(g, args.test_frac, args.split_seed)
  out = {}
  for fam, keep in FEATURE_FAMILIES.items():
    idx = [i for i, nm in enumerate(names) if keep(nm)]
    sub = [names[i] for i in idx]
    out[fam] = fit_logistic(x_all[tr][:, idx], y[tr], x_all[te][:, idx], y[te],
                            sub, args.split_seed)['test_auc']
  return out


def build_parser():
  p = argparse.ArgumentParser(
      description='Stage 2.5 boundary-shortcut audit. Fits boundary-only '
                  'probes; does NOT train the Stage-3 discriminator.')
  p.add_argument('--flow-config', required=True)
  p.add_argument('--flow-ckpt', action='append', required=True,
                 help='label=path (repeatable)')
  p.add_argument('--crl-ckpt', required=True)
  p.add_argument('--n-contexts', type=int, default=8192)
  p.add_argument('--seed', type=int, default=0, help='diagnostic seed')
  p.add_argument('--split-seed', type=int, default=0,
                 help='deterministic classifier train/test split seed')
  p.add_argument('--test-frac', type=float, default=0.3)
  p.add_argument('--out-dir',
                 default='artifacts/propensity_flow/boundary_shortcut_audit')
  return p


def main(argv=None):
  args = build_parser().parse_args(argv)
  with open(args.flow_config) as f:
    meta = json.load(f)

  ds = BehaviorDataset(meta['dataset_path'], val_frac=meta['val_frac'],
                       seed=meta['split_seed'], state_mode='obs',
                       split_level=meta['split_level'])
  if ds.fingerprint['sha256'] != meta['dataset_sha256']:
    print('ABORT: dataset sha256 mismatch.')
    return 1
  A, C = ds.action_dim, ds.context_dim

  # ---- one cached held-out bank ------------------------------------------- #
  rng = np.random.default_rng(args.seed)
  val_idx = ds._indices('val')                                # noqa: SLF001
  n = min(args.n_contexts, val_idx.size)
  rows = rng.choice(val_idx, size=n, replace=False)
  ctx = jnp.asarray(ds._state[rows])                          # noqa: SLF001
  a_real = np.asarray(ds._action[rows])                       # noqa: SLF001
  groups = np.asarray(ds._episode_of_row[rows])               # noqa: SLF001
  z_flow = jnp.asarray(rng.standard_normal((n, A), dtype=np.float32))
  eps_crl = jnp.asarray(rng.standard_normal((n, A), dtype=np.float32))

  print('=' * 80)
  print('propensity.audit_boundary_shortcut -- Stage 2.5 (no discriminator)')
  print('=' * 80)
  print(f'dataset        {os.path.basename(meta["dataset_path"])}')
  print(f'held-out       {n} validation contexts from '
        f'{len(np.unique(groups))} validation episodes')
  print(f'diagnostic seed {args.seed} | classifier split seed '
        f'{args.split_seed} (grouped by EPISODE, test_frac={args.test_frac})')
  print(f'flow sampler   N={MAIN_N}, per-step clip (unchanged)')
  print('offline only   no environment constructed, no rollout generated')

  # ---- BehaviorFlow actions ------------------------------------------------ #
  fc = meta['flow_config']
  fnet = flow_mod.make_flow_network(flow_mod.FlowConfig(
      context_dim=fc['context_dim'], action_dim=fc['action_dim'],
      hidden_sizes=tuple(fc['hidden_sizes']),
      time_features=fc['time_features'], time_max_freq=fc['time_max_freq'],
      use_layer_norm=fc['use_layer_norm']))
  flows = {}
  for spec in args.flow_ckpt:
    label, _, path = spec.partition('=')
    _, fparams, _ = ckpt_mod.load_checkpoint(path)
    fn = jax.jit(lambda c, z, p=fparams: flow_mod.sample_actions_from_noise(
        fnet.apply, p, c, z, num_steps=MAIN_N, clip=True))
    flows[label] = np.asarray(fn(ctx, z_flow))
    print(f'  flow[{label}] <- {path}')

  # ---- CRL target actions -------------------------------------------------- #
  from crl import checkpoint as crl_ckpt
  from crl import networks as crl_networks
  crl_step, crl_state = crl_ckpt.load_checkpoint(args.crl_ckpt)
  cnets = crl_networks.make_networks(
      obs_dim=meta['state_dim'], goal_dim=meta['goal_dim'], action_dim=A,
      repr_dim=CRL_REPR_DIM, repr_norm=False, repr_norm_temp=True,
      hidden_layer_sizes=CRL_HIDDEN, twin_q=CRL_TWIN_Q,
      use_image_obs=False, use_layer_norm=CRL_LAYER_NORM)
  pparams = crl_state.policy_params
  w0 = np.asarray(pparams['mlp/~/linear_0']['w'])
  assert w0.shape == (C, CRL_HIDDEN[0]), (
      f'CRL actor input width {w0.shape} != context dim {C}')

  @jax.jit
  def crl_params_fn(obs):
    return cnets.policy_network.apply(pparams, obs)

  tp = crl_params_fn(ctx)
  a_crl_mode = np.asarray(jnp.tanh(tp.loc))
  a_crl_sample = np.asarray(jnp.tanh(tp.loc + tp.scale * eps_crl))
  print(f'  crl        <- {args.crl_ckpt}  (step {crl_step}, tanh-squashed, '
        f'no clipping)')
  print(f'  crl pre-tanh |loc| mean {float(jnp.abs(tp.loc).mean()):.3f}  '
        f'max {float(jnp.abs(tp.loc).max()):.3f} | scale mean '
        f'{float(tp.scale.mean()):.4f}')

  # Sensitivity check: CRL was trained on RELABELED goals (full 29-dim future
  # states), not the zero-padded g_cmd used as context here. Re-evaluate its
  # saturation under an in-distribution goal to see whether the stored-goal
  # input is distorting its action statistics. Marginal statistics only -- this
  # does NOT enter any classification (it breaks the shared-context pairing).
  jrows = rng.integers(0, ds.n_transitions, size=n)
  relabel_goal = np.asarray(ds._state[jrows][:, :meta['state_dim']])  # noqa: SLF001
  ctx_relabel = jnp.asarray(np.concatenate(
      [np.asarray(ds._state[rows])[:, :meta['state_dim']],   # noqa: SLF001
       relabel_goal], axis=1))
  tp_rel = crl_params_fn(ctx_relabel)
  a_crl_relabel = np.asarray(jnp.tanh(tp_rel.loc + tp_rel.scale * eps_crl))

  # ---- population statistics ---------------------------------------------- #
  pops = {'real': a_real, 'crl_sample': a_crl_sample, 'crl_mode': a_crl_mode,
          'crl_sample_relabeled_goal_SENSITIVITY': a_crl_relabel}
  for label, a in flows.items():
    pops[f'flow_{label}'] = a
  stats = {k: population_stats(v) for k, v in pops.items()}

  print()
  print('POPULATION BOUNDARY STATISTICS')
  print(f'{"population":<40} {"P(any=1)":>9} {"meanCount":>10} '
        f'{"P(|a|=1)":>9} {"P(>=.99)":>9} {"P(>=.95)":>9} {"meanMinDist":>12}')
  for k, s in stats.items():
    print(f'{k:<40} {s["p_any_exact_boundary"]:>9.4f} '
          f'{s["mean_count_exact_boundary"]:>10.4f} '
          f'{s["p_exact_overall"]:>9.4f} {s["p_ge099_overall"]:>9.4f} '
          f'{s["p_ge095_overall"]:>9.4f} '
          f'{s["mean_min_dist_to_boundary"]:>12.6f}')

  print()
  print('PER-DIMENSION P(|a_j| = 1)')
  keys = ['real'] + [f'flow_{l}' for l in flows] + ['crl_sample', 'crl_mode']
  print(f'{"dim":>4} ' + ' '.join(f'{k:>22}' for k in keys))
  for j in range(A):
    print(f'{j:>4} ' + ' '.join(
        f'{stats[k]["p_exact_per_dim"][j]:>22.4f}' for k in keys))

  # ---- three comparisons --------------------------------------------------- #
  comparisons = []
  for label, a in flows.items():
    comparisons.append(compare(f'A_flow{label}_vs_crl', a, a_crl_sample,
                               groups, args))
  comparisons.append(compare('B_real_vs_crl', a_real, a_crl_sample, groups,
                             args))
  for label, a in flows.items():
    comparisons.append(compare(f'C_flow{label}_vs_real', a, a_real, groups,
                               args))
  # Secondary: against the deterministic eval action.
  for label, a in flows.items():
    comparisons.append(compare(f'A2_flow{label}_vs_crlmode', a, a_crl_mode,
                               groups, args))
  comparisons.append(compare('B2_real_vs_crlmode', a_real, a_crl_mode, groups,
                             args))

  print()
  print('BOUNDARY-ONLY CLASSIFICATION  (features: action boundary/saturation '
        'ONLY -- no state, goal, index, timestep or provenance)')
  hdr = (f'{"comparison":<26} {"posAnyB":>8} {"negAnyB":>8} {"anyAUC":>7} '
         f'{"cntAUC":>7} {"minDAUC":>8} {"logitAUC":>9} {"balAcc":>7} '
         f'{"acc":>7}')
  print(hdr)
  print('-' * len(hdr))
  for c in comparisons:
    s, l = c['scalar_auc'], c['logistic']
    print(f'{c["comparison"]:<26} {c["pos_p_any_boundary"]:>8.4f} '
          f'{c["neg_p_any_boundary"]:>8.4f} {s["any_boundary"]:>7.4f} '
          f'{s["count_boundary"]:>7.4f} {s["min_dist_to_boundary"]:>8.4f} '
          f'{l["test_auc"]:>9.4f} {l["test_balanced_accuracy"]:>7.4f} '
          f'{l["test_accuracy"]:>7.4f}')

  print()
  print('FEATURE-FAMILY ABLATION  (logistic test AUC per family)')
  ablations = {}
  ab_pairs = [(f'flow{l}_vs_crl', a, a_crl_sample) for l, a in flows.items()]
  ab_pairs += [('real_vs_crl', a_real, a_crl_sample)]
  ab_pairs += [(f'flow{l}_vs_real', a, a_real) for l, a in flows.items()]
  fams = list(FEATURE_FAMILIES)
  print(f'{"pair":<24} ' + ' '.join(f'{f:>22}' for f in fams))
  for nm, pos, neg in ab_pairs:
    ablations[nm] = feature_ablation(pos, neg, groups, args)
    print(f'{nm:<24} ' + ' '.join(f'{ablations[nm][f]:>22.4f}' for f in fams))

  print()
  print('DOMINANT SHORTCUT FEATURES (logistic |weight|, top 6)')
  for c in comparisons:
    top = c['logistic']['coefficients'][:6]
    print(f'  {c["comparison"]:<26} ' +
          ' '.join(f'{t["feature"]}={t["weight"]:+.2f}' for t in top))

  # ---- artifacts ----------------------------------------------------------- #
  os.makedirs(args.out_dir, exist_ok=True)
  summary = {
      'audit': 'boundary_shortcut', 'stage': 2.5,
      'discriminator_trained': False,
      'classifier_inputs': 'action boundary/saturation features ONLY',
      'state_or_goal_used_by_classifier': False,
      'environment_rollout_used': False,
      'dataset': meta['dataset_path'],
      'dataset_sha256': meta['dataset_sha256'],
      'n_contexts': int(n),
      'n_val_episodes': int(len(np.unique(groups))),
      'diagnostic_seed': args.seed, 'split_seed': args.split_seed,
      'test_frac': args.test_frac,
      'grouping': 'episode-level (no context or same-episode neighbor spans '
                  'the classifier train/test split)',
      'flow_sampler': {'flow_steps': MAIN_N, 'per_step_clip': [-1.0, 1.0]},
      'flow_checkpoints': {s.partition('=')[0]: s.partition('=')[2]
                           for s in args.flow_ckpt},
      'crl_checkpoint': os.path.abspath(args.crl_ckpt),
      'crl_step': int(crl_step),
      'crl_selection_rationale':
          'final.pkl: best.pkl is selected by environment-rollout success, '
          'which notes/continuous_manski_summary_0716.md flags as offline '
          'model-selection leakage; dataset sha256 uniquely identifies this run',
      'crl_action_range': 'tanh squash (no clipping); float32 tanh can still '
                          'return exactly +/-1 for |pre-tanh| >~ 9',
      'crl_goal_caveat':
          'CRL was trained with relabeled goals (goal_indices=range(29), full '
          'future states); the shared context here supplies the zero-padded '
          'g_cmd, which is off its training distribution. See the '
          '*_relabeled_goal_SENSITIVITY population.',
      'populations': stats,
      'comparisons': comparisons,
      'feature_family_ablation': ablations,
      'ablation_reading':
          'exact-boundary family dominant => hard-clipping artifact; '
          'near-boundary family dominant => the learned density is wrong in '
          'the shell just inside the box, which removing the clip would not fix',
  }
  with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)
  with open(os.path.join(args.out_dir, 'classifier_metrics.json'), 'w') as f:
    json.dump(comparisons, f, indent=2)
  with open(os.path.join(args.out_dir, 'per_dimension.csv'), 'w',
            newline='') as f:
    w = csv.writer(f)
    w.writerow(['population', 'dim', 'p_exact_boundary', 'p_ge_0.99',
                'p_ge_0.95', 'mean', 'std'])
    for k, s in stats.items():
      for j in range(A):
        w.writerow([k, j, s['p_exact_per_dim'][j], s['p_ge099_per_dim'][j],
                    s['p_ge095_per_dim'][j], s['mean_per_dim'][j],
                    s['std_per_dim'][j]])
  print(f'\nartifacts -> {os.path.abspath(args.out_dir)}')
  return 0


if __name__ == '__main__':
  sys.exit(main())
