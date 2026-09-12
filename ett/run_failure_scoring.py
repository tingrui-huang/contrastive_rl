"""Scope-B privileged-label feasibility probe; never optimizes ETT or policies."""
import argparse
import json
from pathlib import Path
import pickle
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from sklearn.neighbors import NearestNeighbors

from ett.dataset import TransitionArrays
from ett.failure_scoring import (SCOPE, FEATURES, FailureScorer,
    visible_features, scorer_episode_split, labels_from_death_rows)
from ett.run_distribution_matching import write_json
from propensity.dataset import BehaviorDataset
from propensity.expert_population import resolve_expert_positive_episodes
from scripts.make_swamp_f4_failure_bank import file_sha
from scripts.diagnose_f4_death_observability import (
    _make_probe, _normalizer, _binary_cross_entropy, _sigmoid,
    _predict_logits, _classification_metrics, _reconstruct_temporal_labels,
    _evaluation_masks, _region_masks, _temporal_audit,
    _episode_bootstrap_difference, _git_provenance)

SOURCE = Path('artifacts/ett_distribution_matching/f4_p30_s01_guarded')
OBJECTIVE = Path('artifacts/ett_objective_diagnostic/f4_p30_s01')


def read(path):
  return json.loads(Path(path).read_text(encoding='utf-8'))


def load_population(path):
  """All observed source populations for supervised probing, not nominal BC."""
  expert_ids, population = resolve_expert_positive_episodes(path)
  behavior = BehaviorDataset(path, seed=0, val_frac=.1, state_mode='obs',
      split_level='episode', split_reference='source', strict_bounds=True)
  if not behavior.check()[0]:
    raise ValueError('source behavior loader contract failed')
  with np.load(path, allow_pickle=False) as data:
    obs, act = data['obs'], data['act']
    lengths = data['lengths'] if 'lengths' in data else np.full(len(obs), obs.shape[1])
    meta = json.loads(str(data['meta']))
  episodes = np.repeat(np.arange(len(obs)), lengths-1)
  rows = np.concatenate([np.arange(length-1) for length in lengths])
  arrays = TransitionArrays(obs[episodes, rows, :8], act[episodes, rows],
      obs[episodes, rows+1, :8], obs[episodes, rows, 8:16], episodes, rows)
  if not np.array_equal(arrays.next_state[:, 2:], arrays.state[:, :6]):
    raise ValueError('F4 history order/transition alignment mismatch')
  if not np.array_equal(np.concatenate([arrays.state, arrays.goal], -1), behavior._state):
    raise ValueError('visible observations differ from established loader')
  if not np.array_equal(arrays.goal, np.broadcast_to(arrays.goal[0], arrays.goal.shape)):
    raise ValueError('this bounded comparison expects the existing constant commanded goal')
  labels = _reconstruct_temporal_labels(path, arrays)
  aligned = labels_from_death_rows(episodes, rows, labels['death_observation_row_by_episode'])
  for key in aligned:
    if not np.array_equal(aligned[key], labels[key]):
      raise ValueError('independent label/time alignment check failed')
  splits = scorer_episode_split(len(obs))
  if not np.array_equal(splits['test'], np.unique(behavior._episode_of_row[behavior._val_idx])):
    raise ValueError('source episode holdout changed')
  # Complete observed trajectories, not repeated stationary rows, define duplicates.
  signatures = {}
  for name, ids in splits.items():
    for ep in ids:
      signature = (obs[ep, :lengths[ep]].tobytes(), act[ep, :lengths[ep]-1].tobytes())
      previous = signatures.setdefault(signature, name)
      if previous != name:
        raise ValueError('byte-identical complete trajectory crosses scorer partitions')
  source = np.where(episodes < expert_ids[0], 'uniform_coverage',
      np.where(episodes <= expert_ids[-1], 'expert_positive_source', 'blind_demonstrator'))
  return arrays, labels, splits, source, meta, population


def take_labels(labels, indices, n):
  return {key: value[indices] for key, value in labels.items() if np.shape(value) == (n,)}


def strata(arrays, labels, source, bank_episodes, bank_vectors, prior_episodes):
  dead = labels['dead_when_observed']
  exact = np.array([row.tobytes() in bank_vectors for row in arrays.state])
  result = {'all': np.ones(len(dead), bool), **_evaluation_masks(arrays, labels)}
  result.update({'region_'+key: value for key,value in _region_masks(arrays.state).items()})
  result.update({'source_'+name: source == name for name in np.unique(source)})
  result.update({'bank_episode': np.isin(arrays.episode_id, bank_episodes),
      'outside_bank_episode': ~np.isin(arrays.episode_id, bank_episodes),
      'exact_bank_vector': exact, 'dead_nonmember_vector': dead & ~exact,
      'dead_unseen_bank_episode': dead & ~np.isin(arrays.episode_id, bank_episodes),
      'previous_objective_development_episode': np.isin(arrays.episode_id, prior_episodes),
      'outside_previous_objective_development_episodes': ~np.isin(arrays.episode_id, prior_episodes),
      'onset_vs_alive': (labels['death_age'] == 0) | ~dead,
      'early_vs_alive': ((labels['death_age'] >= 0) & (labels['death_age'] <= 2)) | ~dead})
  waiting = ~dead & labels['zero_action']
  near_goal = np.linalg.norm(arrays.state[:,:2]-arrays.goal[:,:2], axis=-1) < .5
  result['alive_zero_action_away_from_goal'] = waiting & ~near_goal
  result['alive_zero_action_near_goal'] = waiting & near_goal
  return result


def episode_summary(value, episodes, mask, reps=500, seed=772):
  """Context-weighted mean with complete-episode bootstrap uncertainty."""
  count = int(mask.sum())
  if not count:
    return {'count': 0, 'episodes': 0, 'mean': None, 'ci95_episode': None}
  ids, inverse = np.unique(episodes[mask], return_inverse=True)
  sums = np.bincount(inverse, weights=np.asarray(value)[mask])
  counts = np.bincount(inverse)
  draws = np.random.default_rng(seed).integers(len(ids), size=(reps, len(ids)))
  estimates = sums[draws].sum(1)/counts[draws].sum(1)
  return {'count': count, 'episodes': len(ids), 'mean': float(np.asarray(value)[mask].mean()),
          'ci95_episode': np.quantile(estimates, [.025, .975]),
          'small_stratum': len(ids) < 20}


def fit_probe(kind, seed, train, selection, train_y, selection_y, args):
  x = np.asarray(visible_features(train.state, train.goal, kind))
  v = np.asarray(visible_features(selection.state, selection.goal, kind))
  mean, std, constant = _normalizer(x)
  x, v = (x-mean)/std, (v-mean)/std
  net = _make_probe((64, 64))
  params = net.init(jax.random.PRNGKey(seed), jnp.asarray(x[:2]))
  optimizer = optax.adam(3e-4)
  opt_state = optimizer.init(params)
  @jax.jit
  def update(p, o, batch, target):
    loss, grad = jax.value_and_grad(lambda q: _binary_cross_entropy(net.apply(q, batch), target))(p)
    updates, o = optimizer.update(grad, o, p)
    return optax.apply_updates(p, updates), o, loss
  def selection_loss(p):
    z = _predict_logits(net, p, v)
    return float(np.mean(np.logaddexp(0, z)-selection_y*z))
  initial = selection_loss(params)
  history = [{'step': 0, 'train_bce': None, 'selection_bce': initial}]
  best, best_step, best_loss = params, 0, initial
  rng = np.random.default_rng(seed)
  started = time.time()
  for step in range(1, args.steps+1):
    indices = rng.integers(len(x), size=args.batch_size)
    params, opt_state, loss = update(params, opt_state, x[indices], train_y[indices].astype(np.float32))
    if not np.isfinite(float(loss)):
      raise RuntimeError('non-finite training loss')
    if step % args.eval_every == 0 or step == args.steps:
      selected = selection_loss(params)
      if not np.isfinite(selected):
        raise RuntimeError('non-finite selection loss')
      history.append({'step': step, 'train_bce': float(loss), 'selection_bce': selected})
      if selected < best_loss:
        best, best_step, best_loss = params, step, selected
  cp = {'scope': SCOPE, 'params': jax.device_get(best), 'feature_name': kind,
        'hidden_sizes': [64,64], 'mean': mean, 'std': std, 'seed': seed,
        'fixed_commanded_goal': train.goal[0], 'best_step': best_step,
        'selection_bce': best_loss, 'supervision': 'existing simulator audit death labels',
        'offline_compatible': False}
  path = Path(args.out_dir)/f'supervised_probe_{kind}_s{seed}.pkl'
  with path.open('wb') as stream:
    pickle.dump(cp, stream, protocol=pickle.HIGHEST_PROTOCOL)
  report = {'scope': SCOPE, 'normalization': {'mean': mean, 'std': std, 'constant': constant,
      'fit_split': 'scorer_train_only'}, 'history': history, 'all_losses_finite': True,
      'best_step': best_step, 'selection_bce': best_loss, 'elapsed_seconds': time.time()-started,
      'parameter_count': sum(v.size for v in jax.tree.leaves(best)),
      'checkpoint': path.as_posix(), 'checkpoint_sha256': file_sha(path)}
  print(f'{kind} seed={seed} steps={args.steps} best={best_step} selection BCE={best_loss:.6f}', flush=True)
  return FailureScorer(cp), report


def predict(model, state):
  if not hasattr(model, '_compiled_score'):
    model._compiled_score = jax.jit(model.score)
  function = model._compiled_score
  return np.concatenate([np.asarray(function(state[start:start+8192])) for start in range(0, len(state), 8192)])


def evaluate_observed(models, arrays, labels, masks, reps):
  y = labels['dead_when_observed']
  predictions, reports = {}, {}
  for name, model in models.items():
    p = predict(model, arrays.state)
    predictions[name] = p
    report = {}
    for group, mask in masks.items():
      if not mask.any():
        report[group] = {'count': 0, 'episodes': 0, 'reason': 'empty stratum, not dropped'}
        continue
      report[group] = {**_classification_metrics(y[mask], p[mask]),
          'score_episode_interval': episode_summary(p, arrays.episode_id, mask),
          'positive_prediction_episode_interval': episode_summary((p >= .5).astype(float), arrays.episode_id, mask),
          'episodes': len(np.unique(arrays.episode_id[mask]))}
    reports[name] = report
  paired = {}
  for seed in sorted({int(name.rsplit('s', 1)[1]) for name in models}):
    for other in ('xy', 'motion'):
      paired[f'f4_minus_{other}_s{seed}'] = {
          group: _episode_bootstrap_difference(y, predictions[f'f4_s{seed}'], predictions[f'{other}_s{seed}'],
              arrays.episode_id, masks[group], reps, 9331)
          for group in ('all', 'region_swamp_corridor', 'onset_vs_alive', 'early_vs_alive')}
  return reports, predictions, paired


def make_support_diagnostic(arrays):
  """Approximate support, fitted within scorer training episodes only."""
  ids = np.unique(arrays.episode_id)
  perm = np.random.default_rng(615).permutation(ids)
  reference_mask = np.isin(arrays.episode_id, perm[:int(.9*len(perm))])
  rng = np.random.default_rng(616)
  reference_ids = rng.choice(np.flatnonzero(reference_mask), 12000, replace=False)
  calibration_ids = rng.choice(np.flatnonzero(~reference_mask), 4000, replace=False)
  mean, std, _ = _normalizer(arrays.state)
  nn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit((arrays.state[reference_ids]-mean)/std)
  def distance(state):
    state = np.asarray(state)
    return nn.kneighbors((state.reshape(-1,8)-mean)/std, return_distance=True)[0][:,0].reshape(state.shape[:-1])
  threshold = float(np.quantile(distance(arrays.state[calibration_ids]), .99))
  return distance, {'definition': 'full-F4 standardized nearest observed training-row distance',
      'reference_rows': reference_ids, 'reference_episode_ids': arrays.episode_id[reference_ids],
      'internal_calibration_rows': calibration_ids,
      'internal_calibration_episode_ids': arrays.episode_id[calibration_ids],
      'threshold': threshold, 'threshold_rule': '99th percentile on disjoint scorer-training episodes',
      'mean': mean, 'std': std,
      'limitation': 'approximate sparse support diagnostic, not a density, label or physical reachability test'}


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--out-dir', required=True)
  parser.add_argument('--steps', type=int, default=2000)
  parser.add_argument('--seeds', default='0,1')
  parser.add_argument('--batch-size', type=int, default=1024)
  parser.add_argument('--eval-every', type=int, default=250)
  parser.add_argument('--bootstrap-replicates', type=int, default=200)
  parser.add_argument('--numerical-only', action='store_true')
  args = parser.parse_args()
  out = Path(args.out_dir)
  if out.exists():
    raise ValueError('use a fresh directory; prior artifacts are preserved')
  out.mkdir(parents=True)
  started = time.time()
  old, diagnostic = read(SOURCE/'config.json'), read(OBJECTIVE/'config.json')
  path = old['arguments']['dataset']
  tracked_inputs = {**diagnostic['input_file_sha256']}
  for root in (SOURCE, OBJECTIVE, Path('artifacts/death_observability/f4_p30_expert_s01')):
    tracked_inputs.update({p.as_posix(): file_sha(p) for p in root.rglob('*') if p.is_file()})
  missing = [p for p in tracked_inputs if not Path(p).is_file()]
  if missing:
    write_json(out/'blocked.json', {'scope': SCOPE, 'missing': missing, 'training_completed': False})
    raise FileNotFoundError(missing)
  for p, expected in tracked_inputs.items():
    if file_sha(p) != expected:
      raise ValueError(f'preserved input has changed: {p}')
  arrays, labels, split, source, meta, population = load_population(path)
  refs = diagnostic['references']
  with np.load(OBJECTIVE/'visualization_samples.npz', allow_pickle=False) as data:
    bank = data['references']
  bank_vectors = {row.tobytes() for row in bank}
  bank_ids = np.asarray(refs['kept_episode_ids'])
  prior_ids = np.unique(diagnostic['evaluation_episode_ids'])
  partitions = {name: np.flatnonzero(np.isin(arrays.episode_id, ids)) for name, ids in split.items()}
  subsets = {name: arrays.take(ids) for name, ids in partitions.items()}
  subset_labels = {name: take_labels(labels, ids, len(arrays.state)) for name, ids in partitions.items()}
  masks = {name: strata(subsets[name], subset_labels[name], source[ids], bank_ids, bank_vectors, prior_ids)
           for name, ids in partitions.items()}
  config = {'scope': SCOPE, 'offline_compatible': False, 'status': 'running',
      'command': 'python -m ett.run_failure_scoring '+' '.join(sys.argv[1:]),
      'arguments': vars(args), 'git': _git_provenance(), 'dataset': meta,
      'source_population_provenance': population, 'input_file_sha256': tracked_inputs,
      'splits': {name: {'episode_ids': ids, 'episodes': len(ids), 'rows': len(partitions[name]),
          'bank_episode_overlap': np.intersect1d(ids, bank_ids),
          'prior_objective_development_episode_overlap': np.intersect1d(ids, prior_ids),
          'temporal_audit': _temporal_audit(subsets[name], subset_labels[name])} for name, ids in split.items()},
      'split_scope': 'source seed-0 10% holdout; remaining episodes split 90/10 using seed 9201; scorer-independent only',
      'upstream_reuse': 'all episodes seen by original actor; expert holdout previously used for observability selection; 427 holdout episodes contain prior objective contexts',
      'inputs': {'xy': 'current XY + commanded goal', 'motion': '[p0-p1,p1-p2,p2-p3] + commanded goal', 'f4': 'newest-first full F4 + commanded goal',
          'goal': 'same existing width-8 constant goal retained in all arms; normalizes to zero',
          'excluded': 'actions, future state, reward, death/swamp fields, source membership, episode/time IDs'},
      'supervision': 'dead_when_observed from existing pre-action swamp masks, episode death audit and fatal next-cell timing',
      'objective': 'unweighted BCE; uniform observed-transition sampling with replacement, no class balancing',
      'budget': {'steps': args.steps, 'batch_size': args.batch_size, 'seeds': args.seeds, 'hidden_sizes': [64,64], 'learning_rate': 3e-4},
      'selection': 'minimum BCE on scorer-selection episodes; test never used for selection; threshold fixed 0.5',
      'runtime': {'jax': jax.__version__, 'devices': list(map(str, jax.devices()))},
      'source_file_sha256': {p: file_sha(p) for p in ('ett/failure_scoring.py','ett/run_failure_scoring.py','ett/audit_failure_scores.py','scripts/test_failure_scoring.py')}}
  write_json(out/'config.json', config)
  models, training = {}, {}
  for kind in FEATURES:
    for seed in map(int, args.seeds.split(',')):
      name = f'{kind}_s{seed}'
      models[name], training[name] = fit_probe(kind, seed, subsets['train'], subsets['selection'],
          subset_labels['train']['dead_when_observed'], subset_labels['selection']['dead_when_observed'], args)
      write_json(out/'training.json', training)
  if not args.numerical_only:
    report, predictions, paired = evaluate_observed(models, subsets['test'], subset_labels['test'], masks['test'], args.bootstrap_replicates)
    write_json(out/'observed_test_metrics.json', {'scope': SCOPE, 'models': report})
    write_json(out/'paired_episode_intervals.json', paired)
    np.savez_compressed(out/'observed_test_predictions.npz', scope=np.array(SCOPE),
        state=subsets['test'].state, goal=subsets['test'].goal,
        episode_id=subsets['test'].episode_id, timestep=subsets['test'].timestep,
        source=source[partitions['test']], audit_dead=subset_labels['test']['dead_when_observed'],
        audit_death_age=subset_labels['test']['death_age'],
        **{'score_'+name: p for name,p in predictions.items()})
    distance, support = make_support_diagnostic(subsets['train'])
    test_distance = distance(subsets['test'].state)
    support['test_ood_fraction'] = float((test_distance > support['threshold']).mean())
    support['test_confident_ood'] = {name: episode_summary((p>.9).astype(float), subsets['test'].episode_id,
        test_distance>support['threshold']) for name,p in predictions.items()}
    write_json(out/'support_diagnostic.json', support)
    from ett.audit_failure_scores import run_candidate_audit
    run_candidate_audit(out, models, arrays, labels, subsets['test'], subset_labels['test'],
        masks['test'], distance, support['threshold'])
  config['status'] = 'complete'
  config['elapsed_seconds'] = time.time()-started
  config['all_previous_artifacts_unchanged'] = all(file_sha(p)==expected for p,expected in tracked_inputs.items())
  if not config['all_previous_artifacts_unchanged']:
    raise RuntimeError('previous artifacts changed')
  write_json(out/'config.json', config)
  print(f'completed scope-B feasibility run in {config["elapsed_seconds"]:.1f}s; no ETT/policy updates', flush=True)


if __name__ == '__main__':
  main()
