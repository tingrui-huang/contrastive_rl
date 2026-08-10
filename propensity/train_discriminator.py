"""Stage 3A trainer: real-positive behavior-vs-target support discriminator.

Positives are REAL offline behavior actions -- no generative model is involved.
BehaviorFlow is deliberately not imported here (the Stage-2.5 audit showed
flow-generated positives carry a ~0.97-AUC boundary source signature, while real
behavior actions carry ~0.52, i.e. chance).

  positive:  (s_i, g_cmd_i, a_real_i)                      -> y = 1
  negative:  (s_i, g_cmd_i, a_crl_i),
             a_crl_i ~ pi(. | s_i, g_query_i)               -> y = 0

``g_query_i`` is drawn with the CRL replay future-goal law and decides only
WHICH action the learner proposes. It is NOT a discriminator input: it is an
achieved future state, hence a descendant of the action, and conditioning a
behavior-support model on it would turn the object into a hindsight posterior.

The trained score is a RELATIVE support/discrepancy score, not a propensity and
not a causal mixture weight. See propensity/discriminator.py.

Splits are episode-level and three-way: Stage-1 validation episodes are held
back as the FINAL TEST set and are never used for model selection; the Stage-1
training episodes are split deterministically into train and dev.

Run:

  python -m propensity.train_discriminator \
      --dataset artifacts/rockfall_v2_p30_h800_resetfix/pilot/antmaze_rockfall_v2_p30_h800_resetfix_pilot.npz \
      --crl-ckpt naive_rockfall_v2_p30_h800_resetfix_s0_300k/final.pkl \
      --input-spec state_cmdgoal_action \
      --out-dir artifacts/support_discriminator/D_state_cmdgoal_action
"""
import argparse
import json
import os
import pickle
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(_HERE) not in sys.path:
  sys.path.insert(0, os.path.dirname(_HERE))

import jax
import jax.numpy as jnp
import numpy as np
import optax

from propensity import crl_policy_adapter as crl_adapter
from propensity import discriminator as disc_mod
from propensity.dataset import BehaviorDataset


def build_parser():
  p = argparse.ArgumentParser(
      description='Stage 3A: train a behavior-vs-target RELATIVE SUPPORT '
                  'discriminator from real offline behavior actions. '
                  'Not a propensity estimator.')
  p.add_argument('--dataset', required=True)
  p.add_argument('--crl-ckpt', required=True)
  p.add_argument('--input-spec', required=True,
                 choices=sorted(disc_mod.INPUT_SPECS))
  p.add_argument('--out-dir', required=True)
  # goal representation
  p.add_argument('--cmdgoal-indices', default='0,1',
                 help='LIVE g_cmd indices for this environment (rockfall: 0,1; '
                      'the other 27 dims are identically zero). The general '
                      'formulation stays D(s, g_cmd, a).')
  # splits
  p.add_argument('--split-seed', type=int, default=0,
                 help='Stage-1 episode split seed (test = Stage-1 val eps)')
  p.add_argument('--val-frac', type=float, default=0.1)
  p.add_argument('--dev-episodes', type=int, default=30)
  p.add_argument('--dev-split-seed', type=int, default=1234)
  # optimization
  p.add_argument('--seed', type=int, default=0)
  p.add_argument('--contexts-per-batch', type=int, default=128,
                 help='each context yields 1 positive + 1 negative, so the '
                      'batch is exactly balanced at 2x this size')
  p.add_argument('--learning-rate', type=float, default=3e-4)
  p.add_argument('--steps', type=int, default=20000)
  p.add_argument('--eval-every', type=int, default=1000)
  p.add_argument('--hidden-sizes', default='256,256')
  p.add_argument('--dev-contexts', type=int, default=4096)
  return p


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


def make_splits(ds, dev_episodes, dev_seed):
  """Three-way EPISODE-level split.

  Stage-1 validation episodes become the final TEST set untouched; the Stage-1
  training episodes are partitioned into train and dev. A context's positive and
  its negative always land in the same split because the split is keyed on the
  context's episode."""
  ep_of_row = ds._episode_of_row                              # noqa: SLF001
  tr_rows = ds._indices('train')                              # noqa: SLF001
  te_rows = ds._indices('val')                                # noqa: SLF001
  tr_eps = np.unique(ep_of_row[tr_rows])
  te_eps = np.unique(ep_of_row[te_rows])
  rng = np.random.default_rng(dev_seed)
  perm = rng.permutation(tr_eps)
  dev_eps = np.sort(perm[:dev_episodes])
  fit_eps = np.sort(perm[dev_episodes:])
  in_dev = np.isin(ep_of_row, dev_eps)
  in_fit = np.isin(ep_of_row, fit_eps)
  rows = {'train': tr_rows[in_fit[tr_rows]],
          'dev': tr_rows[in_dev[tr_rows]],
          'test': te_rows}
  eps = {'train': fit_eps, 'dev': dev_eps, 'test': te_eps}
  assert len(np.intersect1d(eps['train'], eps['dev'])) == 0
  assert len(np.intersect1d(eps['train'], eps['test'])) == 0
  assert len(np.intersect1d(eps['dev'], eps['test'])) == 0
  return rows, eps


def main(argv=None):
  args = build_parser().parse_args(argv)
  t0 = time.time()
  gidx = [int(x) for x in args.cmdgoal_indices.split(',') if x != '']

  ds = BehaviorDataset(args.dataset, val_frac=args.val_frac,
                       seed=args.split_seed, state_mode='obs',
                       split_level='episode')
  ok, gates, details = ds.check()
  if not ok:
    print('Stage-1 dataset gates FAILED:', gates)
    return 1
  S, A = ds._obs_dim, ds.action_dim                           # noqa: SLF001
  G = len(gidx)

  # dataset sha256 must match the CRL run's recorded training dataset
  crl_side = os.path.join(os.path.dirname(args.crl_ckpt),
                          'offline_dataset.sha256')
  crl_ds_sha = None
  if os.path.exists(crl_side):
    with open(crl_side) as f:
      crl_ds_sha = json.load(f)['sha256']
    if crl_ds_sha != ds.fingerprint['sha256']:
      print('ABORT: CRL run was trained on a different dataset.')
      return 1

  rows, eps = make_splits(ds, args.dev_episodes, args.dev_split_seed)
  lengths = ds._lengths                                       # noqa: SLF001
  ep_of_row = ds._episode_of_row                              # noqa: SLF001
  # timestep within episode, needed for the future-goal law
  starts = np.concatenate([[0], np.cumsum(lengths - 1)])
  t_of_row = np.arange(ds.n_transitions) - starts[ep_of_row]

  crl_step, crl_apply, crl_info = crl_adapter.load_frozen_crl_actor(
      args.crl_ckpt, S, ds._goal_dim, A)                      # noqa: SLF001
  goal_src = crl_adapter.load_goal_source(args.dataset, S)    # [E, L, S]

  print('=' * 78)
  print('propensity.train_discriminator -- Stage 3A relative support score')
  print('=' * 78)
  print(f'input spec       {args.input_spec}  '
        f'-> {disc_mod.INPUT_SPECS[args.input_spec]}')
  print(f'g_cmd indices    {gidx}  (live dims for this env; general form is '
        f'D(s, g_cmd, a))')
  print(f'dataset          {os.path.basename(args.dataset)}  '
        f'sha {ds.fingerprint["sha256"][:16]}...')
  print(f'CRL actor        {args.crl_ckpt} @ step {crl_step} (FROZEN)')
  print(f'episodes         train {len(eps["train"])} | dev {len(eps["dev"])} '
        f'| test {len(eps["test"])}  (test = Stage-1 val, never selected on)')
  print(f'transitions      train {len(rows["train"])} | dev {len(rows["dev"])} '
        f'| test {len(rows["test"])}')
  print('positives        REAL offline behavior actions (no BehaviorFlow)')
  print('negatives        pi(.|s, g_query), g_query ~ CRL replay future law')
  print('NOTE             output is a RELATIVE support score, NOT a propensity')

  # ---- cached dev bank (fixed; used only for model selection) -------------- #
  rng = np.random.default_rng(args.seed)
  dv = rows['dev'][rng.choice(len(rows['dev']),
                              size=min(args.dev_contexts, len(rows['dev'])),
                              replace=False)]
  dev_j = crl_adapter.sample_future_goal_index(
      lengths, ep_of_row[dv], t_of_row[dv], np.random.default_rng(args.seed + 5))
  raw = ds._state                                             # noqa: SLF001
  dev_s = np.asarray(raw[dv][:, :S])
  dev_g = np.asarray(raw[dv][:, S:][:, gidx])
  dev_a_real = np.asarray(ds._action[dv])                     # noqa: SLF001
  # g_query = the ACHIEVED state at the sampled future index. Looked up in the
  # episode-indexed tensor because index L_e-1 (terminal obs) is selectable and
  # has no flattened transition row.
  dev_gq = np.asarray(goal_src[ep_of_row[dv], dev_j])
  dev_a_crl, _ = crl_adapter.crl_actions(
      crl_apply, dev_s, dev_gq,
      np.random.default_rng(args.seed + 6).standard_normal((len(dv), A)))

  # ---- model -------------------------------------------------------------- #
  hidden = tuple(int(x) for x in args.hidden_sizes.split(',') if x)
  in_dim = disc_mod.input_dim_for(args.input_spec, S, G, A)
  cfg = disc_mod.DiscriminatorConfig(input_dim=in_dim, hidden_sizes=hidden)
  net = disc_mod.make_discriminator(cfg)

  # standardizer fitted on TRAIN-split positives+negatives only
  fit_rows = rows['train'][np.random.default_rng(args.seed + 7).choice(
      len(rows['train']), size=min(8192, len(rows['train'])), replace=False)]
  fit_j = crl_adapter.sample_future_goal_index(
      lengths, ep_of_row[fit_rows], t_of_row[fit_rows],
      np.random.default_rng(args.seed + 8))
  fit_s = np.asarray(raw[fit_rows][:, :S])
  fit_g = np.asarray(raw[fit_rows][:, S:][:, gidx])
  fit_gq = np.asarray(goal_src[ep_of_row[fit_rows], fit_j])
  fit_acrl, _ = crl_adapter.crl_actions(
      crl_apply, fit_s, fit_gq,
      np.random.default_rng(args.seed + 9).standard_normal((len(fit_rows), A)))
  fit_x = np.concatenate([
      np.asarray(disc_mod.assemble_inputs(args.input_spec, fit_s, fit_g,
                                          np.asarray(ds._action[fit_rows]))),
      np.asarray(disc_mod.assemble_inputs(args.input_spec, fit_s, fit_g,
                                          fit_acrl))], axis=0)
  std = disc_mod.fit_standardizer(fit_x)

  key = jax.random.PRNGKey(args.seed)
  key, ik = jax.random.split(key)
  params = net.init(ik, jnp.zeros((1, in_dim), jnp.float32))
  n_params = int(sum(np.prod(p.shape)
                     for p in jax.tree_util.tree_leaves(params)))
  opt = optax.adam(args.learning_rate)
  opt_state = opt.init(params)
  print(f'architecture     MLP{hidden} -> 1 logit | input_dim {in_dim} | '
        f'params {n_params}')
  print(f'optimizer        adam(lr={args.learning_rate}) | balanced batch '
        f'{2 * args.contexts_per_batch} ({args.contexts_per_batch} contexts)')
  print()

  @jax.jit
  def update(params, opt_state, x_pos, x_neg):
    def loss_fn(p):
      lp = net.apply(p, std.apply(x_pos))
      ln = net.apply(p, std.apply(x_neg))
      logits = jnp.concatenate([lp, ln])
      labels = jnp.concatenate([jnp.ones_like(lp), jnp.zeros_like(ln)])
      return disc_mod.bce_with_logits(logits, labels)
    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = opt.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), opt_state, loss

  @jax.jit
  def logits_of(params, x):
    return net.apply(params, std.apply(x))

  def dev_metrics(params):
    xp = disc_mod.assemble_inputs(args.input_spec, dev_s, dev_g, dev_a_real)
    xn = disc_mod.assemble_inputs(args.input_spec, dev_s, dev_g, dev_a_crl)
    lp, ln = np.asarray(logits_of(params, xp)), np.asarray(logits_of(params, xn))
    sc = np.concatenate([lp, ln])
    y = np.concatenate([np.ones(len(lp), int), np.zeros(len(ln), int)])
    return _roc_auc(sc, y), float(disc_mod.bce_with_logits(sc, y))

  # ---- train --------------------------------------------------------------- #
  batch_rng = np.random.default_rng(args.seed + 100)
  gq_rng = np.random.default_rng(args.seed + 200)
  eps_rng = np.random.default_rng(args.seed + 300)
  tr = rows['train']
  history, best = [], {'dev_auc': -1.0, 'step': 0, 'params': params}
  print(f'{"step":>7} {"train_bce":>10} {"dev_auc":>9} {"dev_bce":>9} '
        f'{"elapsed_s":>10}')
  running, rn = 0.0, 0
  for step in range(1, args.steps + 1):
    b = tr[batch_rng.integers(0, len(tr), size=args.contexts_per_batch)]
    j = crl_adapter.sample_future_goal_index(lengths, ep_of_row[b],
                                             t_of_row[b], gq_rng)
    s = np.asarray(raw[b][:, :S])
    g = np.asarray(raw[b][:, S:][:, gidx])
    a_real = np.asarray(ds._action[b])                        # noqa: SLF001
    gq = np.asarray(goal_src[ep_of_row[b], j])
    a_crl, _ = crl_adapter.crl_actions(
        crl_apply, s, gq, eps_rng.standard_normal((len(b), A)))
    xp = disc_mod.assemble_inputs(args.input_spec, s, g, a_real)
    xn = disc_mod.assemble_inputs(args.input_spec, s, g, a_crl)
    params, opt_state, loss = update(params, opt_state, xp, xn)
    running += float(loss)
    rn += 1
    if step % args.eval_every == 0 or step == args.steps:
      auc, bce = dev_metrics(params)
      tb = running / max(rn, 1)
      running, rn = 0.0, 0
      history.append({'step': step, 'train_bce': tb, 'dev_auc': auc,
                      'dev_bce': bce, 'elapsed_s': round(time.time() - t0, 2)})
      print(f'{step:>7} {tb:>10.5f} {auc:>9.5f} {bce:>9.5f} '
            f'{time.time() - t0:>10.1f}', flush=True)
      if auc > best['dev_auc']:                 # selection on DEV only
        best = {'dev_auc': auc, 'step': step,
                'params': jax.tree_util.tree_map(np.asarray, params)}

  os.makedirs(args.out_dir, exist_ok=True)
  with open(os.path.join(args.out_dir, 'model.pkl'), 'wb') as f:
    pickle.dump({'step': best['step'], 'params': best['params'],
                 'standardizer': {'mean': std.mean, 'std': std.std}}, f)
  meta = {
      'stage': '3A', 'object': 'relative_support_score',
      'NOT_a_propensity': True,
      'score_semantics':
          'balanced behavior-vs-target classifier; an ideal model recovers '
          'p_behavior/(p_behavior + p_target) under an ARTIFICIAL 50/50 class '
          'prior. sigmoid(logit) is NOT P(A=a|S=s,G=g) and is NOT a causal '
          'branch probability or mixture weight.',
      'behaviorflow_used': False,
      'environment_rollouts_used': False,
      'dataset_path': os.path.abspath(args.dataset),
      'dataset_sha256': ds.fingerprint['sha256'],
      'crl_dataset_sha256': crl_ds_sha,
      'crl': crl_info,
      'query_goal_semantics': crl_adapter.replay_semantics(),
      'crl_negative_sampling': 'one fixed STOCHASTIC sample per context '
                               '(tanh(loc + scale*eps)); mode is a secondary '
                               'sensitivity check at eval time',
      'input_spec': args.input_spec,
      'input_parts': list(disc_mod.INPUT_SPECS[args.input_spec]),
      'general_formulation': 'D(s, g_cmd, a)',
      'cmdgoal_indices': gidx,
      'cmdgoal_note': 'rockfall g_cmd has only dims [0,1] live; the remaining '
                      '27 are identically zero. Narrowing the input to the '
                      'live dims is environment-specific, not a redefinition '
                      'of the general method.',
      'state_dim': S, 'cmdgoal_dim': G, 'action_dim': A, 'input_dim': in_dim,
      'architecture': cfg.asdict(), 'n_params': n_params,
      'learning_rate': args.learning_rate,
      'contexts_per_batch': args.contexts_per_batch,
      'balanced_batch_size': 2 * args.contexts_per_batch,
      'steps': args.steps, 'seed': args.seed,
      'stage1_split_seed': args.split_seed, 'val_frac': args.val_frac,
      'dev_split_seed': args.dev_split_seed,
      'episodes': {k: [int(x) for x in v] for k, v in eps.items()},
      'n_transitions': {k: int(len(v)) for k, v in rows.items()},
      'model_selection': 'highest DEV auc; TEST episodes never used',
      'best_dev_auc': best['dev_auc'], 'best_step': best['step'],
      'standardizer_fitted_on': 'train split only',
      'wall_clock_s': round(time.time() - t0, 2),
  }
  with open(os.path.join(args.out_dir, 'config.json'), 'w') as f:
    json.dump(meta, f, indent=2)
  with open(os.path.join(args.out_dir, 'metrics.json'), 'w') as f:
    json.dump(history, f, indent=2)
  print()
  print(f'best dev AUC {best["dev_auc"]:.5f} @ step {best["step"]}')
  print(f'artifacts -> {os.path.abspath(args.out_dir)}')
  return 0


if __name__ == '__main__':
  sys.exit(main())
