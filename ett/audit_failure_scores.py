"""Frozen scope-B scorer audits on unlabelled synthetic observations."""
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from ett.diagonal_transition import DiagonalTransitionSpec, _project_samples
from ett.diagnose_failure_objectives import geometry, group_summary
from ett.failure_objectives import nearest_reference, shift_with_newest
from ett.run_distribution_matching import write_json
from ett.failure_scoring import SCOPE
from ett.run_failure_scoring import OBJECTIVE, episode_summary, predict, take_labels


def saved_candidates(data):
  """Reconstruct the unchanged objective-diagnostic candidates exactly."""
  s = jnp.asarray(data['state'])
  count = data['primary'].shape[1]
  spec = DiagonalTransitionSpec()
  persistence = np.asarray(shift_with_newest(s, jnp.repeat(s[:,None,:2], count, 1)))
  candidates = {'persistence': persistence, 'diagonal_only': data['control'], 'mmd_trained': data['primary']}
  for epsilon in (.01, .05):
    for name, delta in (('right',[1,0]), ('left',[-1,0]), ('up',[0,1]), ('down',[0,-1])):
      y, _ = _project_samples(s, jnp.broadcast_to(epsilon*jnp.array(delta), (len(s),count,2)), spec)
      candidates[f'shift_{name}_{epsilon:g}'] = np.asarray(y)
  angles = np.arange(count)*2*np.pi/count
  delta = .05*np.stack([np.cos(angles),np.sin(angles)],-1)
  candidates['symmetric_jitter_radius_0.05'] = np.asarray(_project_samples(s, jnp.broadcast_to(delta,(len(s),count,2)),spec)[0])
  details = nearest_reference(jnp.asarray(persistence), jnp.asarray(data['references']), data['mean'], data['std'])
  xy = data['references'][np.asarray(details['history_nearest_index']),:2]
  candidates['history_nearest_collapse_raw'] = np.asarray(shift_with_newest(s,jnp.asarray(xy)))
  candidates['history_nearest_collapse_projected'] = np.asarray(_project_samples(s,jnp.asarray(xy)-s[:,None,:2],spec)[0])
  return candidates


def observed_examples(models, arrays, labels, masks, full_arrays, distance, threshold):
  """Perturb observed scored y, with its OLD frames fixed and labels separate."""
  groups = ('dead_established_age_ge_3', 'dead_age_0_first_observation',
            'dead_age_1', 'alive_zero_recorded_action_waiting',
            'alive_nonzero_action_near_stationary_collision_compatible',
            'alive_reset_seeded_equal_stack')
  lookup = {(int(e),int(t)): i for i,(e,t) in enumerate(zip(full_arrays.episode_id,full_arrays.timestep))}
  selected = []
  for name in groups:
    indices = np.flatnonzero(masks[name])
    # One representative per distinct episode, selected without score values.
    _, first = np.unique(arrays.episode_id[indices], return_index=True)
    indices = indices[first]
    selected.extend((name, int(i)) for i in indices[:8])
  offsets = np.stack(np.meshgrid(np.linspace(-.25,.25,21),np.linspace(-.25,.25,21)), -1).reshape(-1,2)
  translations = np.array([[0,0],[1,0],[-1,0],[0,1],[0,-1]],np.float32)
  functions = {name: jax.jit(model.score) for name, model in models.items()}
  derivatives = {name: jax.jit(jax.grad(model.score)) for name, model in models.items()}
  results, plot_arrays = [], {}
  for number, (group, index) in enumerate(selected):
    y = arrays.state[index].copy()
    ep, t = int(arrays.episode_id[index]), int(arrays.timestep[index])
    prev = full_arrays.state[lookup[(ep,t-1)]] if t else y.copy()
    candidates = np.broadcast_to(y,(len(offsets),8)).copy()
    candidates[:,:2] += offsets
    geo = geometry(prev[None],candidates[None])
    xy = candidates[:,:2]
    # Geometry is tested per candidate, with no claim of physical reachability.
    from ett.eval_diagonal_transition import _open_endpoint
    segment = prev[:2]+np.linspace(0,1,41)[None,:,None]*(xy-prev[:2])[:,None]
    valid = _open_endpoint(xy) & _open_endpoint(segment).all(1) & (np.abs(xy-prev[:2])<=1.+1e-6).all(1)
    values = {'scope': SCOPE, 'group': group, 'episode_id': ep, 'observation_row': t,
        'observed_state': y, 'preceding_context': prev, 'observed_audit_dead': bool(labels['dead_when_observed'][index]),
        'observed_death_age': int(labels['death_age'][index]),
        'synthetic_labels': None, 'grid_half_width_maze_units': .25,
        'grid_geometry_summary': {k:float(v[0]) for k,v in geo.items()},
        'anchor_bound': 'not available for this observed-state/action pair; not asserted', 'models': {}}
    for name, model in models.items():
      fn = functions[name]
      base = float(fn(y))
      gradient = np.asarray(derivatives[name](jnp.asarray(y)))[:2]
      norm = np.linalg.norm(gradient)
      raw = y.copy()
      raw[:2] += .05*gradient/max(norm,1e-12)
      projected = np.asarray(_project_samples(jnp.asarray(prev[None]),
          jnp.asarray((raw[:2]-prev[:2])[None,None]),DiagonalTransitionSpec())[0])[0,0]
      if not np.array_equal(projected[2:], y[2:]):
        raise RuntimeError('gradient candidate changed old frames')
      score = np.asarray(fn(candidates))
      winner = int(np.argmax(np.where(valid,score,-np.inf)))
      shifted = y[None]+np.tile(translations,(1,4))
      translated_score = np.asarray(fn(shifted))
      grad_geo = geometry(prev[None],projected[None,None])
      values['models'][name] = {'base_score': base, 'newest_xy_gradient': gradient,
          'newest_xy_gradient_norm': float(norm), 'projected_gradient_step_score': float(fn(projected)),
          'projected_gradient_step_motion_from_observed': float(np.linalg.norm(projected[:2]-y[:2])),
          'projected_gradient_step_geometry': {k:float(v[0]) for k,v in grad_geo.items()},
          'grid_best_score': float(score[winner]), 'grid_score_gain': float(score[winner]-base),
          'grid_best_xy': candidates[winner,:2], 'grid_best_motion_from_observed': float(np.linalg.norm(offsets[winner])),
          'grid_best_support_distance': float(distance(candidates[winner])),
          'grid_best_ood': bool(distance(candidates[winner])>threshold),
          'whole_F4_translation_scores': translated_score,
          'whole_F4_translation_score_range': float(np.ptp(translated_score)),
          'translation_scope': 'location sensitivity only; not a ground-truth-preserving transformation'}
      if number % 8 == 0 and name == 'f4_s0':
        plot_arrays[f'score_{number}'] = score
    if number % 8 == 0:
      plot_arrays[f'xy_{number}'] = xy
      plot_arrays[f'valid_{number}'] = valid
    results.append(values)
  return results, plot_arrays


def run_candidate_audit(out, models, full_arrays, full_labels, test, test_labels,
                        test_masks, distance, threshold):
  with np.load(OBJECTIVE/'visualization_samples.npz',allow_pickle=False) as npz:
    data = {key:npz[key] for key in npz.files}
  lookup = {(int(e),int(t)):i for i,(e,t) in enumerate(zip(full_arrays.episode_id,full_arrays.timestep))}
  indices = np.array([lookup[(int(e),int(t))] for e,t in zip(data['episode_id'],data['timestep'])])
  if not np.array_equal(full_arrays.state[indices],data['state']):
    raise ValueError('saved evaluation contexts differ from source states')
  labels = take_labels(full_labels,indices,len(full_arrays.state))
  groups = {'all': np.ones(len(indices),bool), 'confirmed_dead_context': labels['dead_when_observed'],
      'confirmed_established_dead_context': labels['death_age']>=3,
      'confirmed_onset_context': labels['death_age']==0,
      'alive_context': ~labels['dead_when_observed'], 'frozen_nonreset_context': data['frozen'],
      'moving_context': data['moving']}
  candidates = saved_candidates(data)
  geometry_results, scores, raw = {}, {}, {}
  for candidate, y in candidates.items():
    geometry_results[candidate] = group_summary(geometry(data['state'],y,data['anchor'],data['radius']),groups)
    support = distance(y)
    scores[candidate], raw[candidate] = {}, {}
    for name, model in models.items():
      p = predict(model,y)
      raw[candidate][name] = p
      scores[candidate][name] = {group: {
          'score': episode_summary(p.mean(1),data['episode_id'],mask),
          'confident_score_fraction': episode_summary((p>.9).mean(1),data['episode_id'],mask),
          'ood_fraction': episode_summary((support>threshold).mean(1),data['episode_id'],mask),
          'confident_ood_fraction': episode_summary(((support>threshold)&(p>.9)).mean(1),data['episode_id'],mask)}
          for group,mask in groups.items()}
  paired = {}
  for candidate in candidates:
    paired[candidate] = {name: {group: episode_summary((raw[candidate][name]-raw['persistence'][name]).mean(1),
        data['episode_id'],mask) for group,mask in groups.items()} for name in models}
  write_json(out/'controlled_candidate_scores.json', {'scope': SCOPE, 'scores':scores, 'versus_persistence':paired,
      'synthetic_ground_truth_labels': None, 'group_labels_apply_to_source_context_only': True,
      'all_1024_contexts_are_previous_development_data': True})
  write_json(out/'controlled_candidate_geometry.json', {'geometry':geometry_results,
      'bound': 'relative to the original saved primary ETT draw anchor/radius; constructed candidates may violate it',
      'reachability': 'endpoint/cap/41-point segment checks are not full simulator reachability'})
  examples, plots = observed_examples(models,test,test_labels,test_masks,full_arrays,distance,threshold)
  write_json(out/'observed_gradient_examples.json', {'scope':SCOPE,'selection':'first eight distinct test episodes per stated stratum, independent of scores','examples':examples})
  # Explicitly invalid and translated F4 probes retain labels as unknown.
  ood = np.concatenate([np.tile(np.array([[x,y]],np.float32),(1,4)) for x in (-2.,.5,3.5,5.5,10.) for y in (-2.,1.5,3.5,7.)])
  from ett.eval_diagonal_transition import _open_endpoint
  write_json(out/'synthetic_sensitivity.json', {'scope':SCOPE, 'ground_truth':None,
      'definition':'constant-stack coordinate sweep includes walls/outside maze; sensitivity only',
      'states':ood, 'endpoint_valid':_open_endpoint(ood[:,:2]), 'support_distance':distance(ood),
      'ood_threshold':threshold, 'scores':{name:predict(model,ood) for name,model in models.items()}})
  np.savez_compressed(out/'gradient_plot_arrays.npz',scope=np.array(SCOPE),**plots)
  print('completed frozen-score candidate, gradient, support and geometry audits',flush=True)
