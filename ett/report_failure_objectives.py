"""Render saved objective diagnostics without resampling or training networks."""
import argparse
import os
from pathlib import Path

os.environ.setdefault('MPLBACKEND', 'Agg')
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from ett.diagnose_failure_objectives import read, geometry
from ett.failure_objectives import nearest_reference, mmd_components, shift_with_newest
from ett.run_distribution_matching import write_json
from scripts.make_swamp_f4_failure_bank import file_sha


PRIMARY = 's0_L0p25_lambda1'
CONTROL = 's0_L0p25_lambda0'


def collapse_probe(data):
  s, f = jnp.asarray(data['state']), jnp.asarray(data['references'])
  mean, std = jnp.asarray(data['mean']), jnp.asarray(data['std'])
  persistence = shift_with_newest(s, s[:, None, :2])
  detail = nearest_reference(persistence, f, mean, std)
  xy = np.asarray(f)[np.asarray(detail['history_nearest_index'])[:, 0], :2]
  candidate = np.asarray(shift_with_newest(s, jnp.repeat(jnp.asarray(xy)[:, None], data['primary'].shape[1], axis=1)))
  feasible = geometry(data['state'], candidate, data['anchor'], data['radius'])
  valid = np.ones(len(s), bool)
  for key in ('endpoint_invalid_fraction', 'segment_blocked_fraction', 'one_step_cap_violation_fraction', 'anchor_bound_violation_fraction'):
    valid &= feasible[key] == 0
  primary = nearest_reference(jnp.asarray(data['primary']), f, mean, std)['distance2'].mean(1)
  collapsed = np.asarray(detail['history_floor'])[:, 0]
  result = {}
  for group, mask in [('all', np.ones(len(s), bool)), ('frozen_nonreset', data['frozen']), ('moving', data['moving'])]:
    subset = mask & valid
    result[group] = {
        'contexts': int(mask.sum()), 'all_sampled_anchors_and_geometry_pass_contexts': int(subset.sum()),
        'feasible_fraction': float(subset.sum()/mask.sum()),
        'mean_collapse_motion_all_contexts': float(feasible['motion'][mask].mean()),
        'primary_set_cost_feasible_subset': float(np.asarray(primary)[subset].mean()) if subset.any() else None,
        'collapsed_set_cost_feasible_subset': float(collapsed[subset].mean()) if subset.any() else None,
        'set_improved_feasible_fraction': float((collapsed[subset] < np.asarray(primary)[subset]-1e-8).mean()) if subset.any() else None,
        'collapsed_xy_std': 0.,
    }
  return {'definition': 'one history-nearest newest position repeated across samples; old frames unchanged',
          'feasibility_scope': 'all 16 SAVED sampled anchors/radii plus endpoint, cap and 41-point segment checks; not an all-randomness bound or physics proof',
          'groups': result}


def plot_decomposition(out, paired):
  fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
  for axis, group in zip(axes, ('all', 'frozen_nonreset', 'moving')):
    values = paired[PRIMARY]['16'][group]
    names = ['gg', 'cross_contribution', 'mmd']
    centers = np.array([values[n]['mean'] for n in names])
    ci = np.array([values[n]['ci95_episode'] for n in names])
    axis.bar(range(3), centers, color=['C0', 'C1', 'C2'])
    axis.errorbar(range(3), centers, yerr=np.maximum(0, np.stack([centers-ci[:, 0], ci[:, 1]-centers])),
                  fmt='none', color='black', capsize=4)
    axis.axhline(0, color='.5', linewidth=.8)
    axis.set_xticks(range(3), ['Change in GG', '-2 change in GR', 'MMD change'], rotation=20)
    axis.set_title(f'{group}\nRR change = 0; K=16')
    axis.set_ylabel('MMD-trained minus diagonal-only')
  fig.savefig(out/'mmd_decomposition.png', dpi=170)
  plt.close(fig)


def plot_k(out, sampled):
  fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
  for axis, group in zip(axes, ('all', 'frozen_nonreset')):
    for name, label, color in [(CONTROL, 'Diagonal-only', 'C0'), (PRIMARY, 'MMD-trained', 'C1')]:
      for metric, marker in [('mmd', 'o-'), ('mmd_generated_u', 's--')]:
        axis.plot([8, 16], [sampled[name][str(k)]['mean_over_sampling_seeds'][group][metric] for k in (8, 16)],
                  marker, color=color, label=label+(' / V' if metric == 'mmd' else ' / generated U'))
    axis.set_xticks([8, 16])
    axis.set(xlabel='Independent expert/transition draws K', ylabel='MMD estimate', title=group)
    axis.legend(fontsize=8)
  fig.savefig(out/'sample_count_bias.png', dpi=170)
  plt.close(fig)


def plot_candidates(out, controlled):
  names = ['persistence', 'diagonal_only', 'mmd_trained', 'symmetric_jitter_radius_0.05', 'history_nearest_collapse_projected']
  labels = ['Persistence', 'Diagonal-only', 'MMD-trained', '0.05 XY jitter', 'Projected collapse']
  fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
  for row, group in enumerate(('frozen_nonreset', 'moving')):
    for col, metric in enumerate(('mmd', 'set_cost')):
      axis = axes[row, col]
      axis.bar(range(len(names)), [controlled['candidates'][name][group][metric] for name in names])
      axis.set_xticks(range(len(names)), labels, rotation=20, ha='right', fontsize=9)
      axis.set_title(group + (' / MMD' if metric == 'mmd' else ' / nearest-set cost'))
      axis.set_ylabel('Lower objective value')
  fig.savefig(out/'controlled_candidates.png', dpi=170)
  plt.close(fig)


def plot_interpolation(out, trajectory):
  fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
  ts = [float(t) for t in trajectory['fractions']]
  for axis, metric in zip(axes, ('mmd', 'set_cost')):
    for group in ('all', 'frozen_nonreset', 'moving'):
      ys = [trajectory['fractions'][str(t)][group][metric] for t in ts]
      axis.plot(ts, np.asarray(ys)-ys[0], 'o-', label=group)
    axis.axhline(0., color='.5', linewidth=.8)
    axis.set(xlabel='Fraction of actual anchor-to-emitted displacement', ylabel=f'Change in {metric}')
    axis.legend(fontsize=8)
  fig.savefig(out/'anchor_displacement_losses.png', dpi=170)
  plt.close(fig)


def plot_landscapes(out, data):
  state, references = data['state'], jnp.asarray(data['references'])
  mean, std, bandwidths = [jnp.asarray(data[n]) for n in ('mean','std','bandwidths')]
  targets = [('frozen', data['frozen'], [3.5,3.5]), ('frozen', data['frozen'], [5.,3.5]),
             ('moving upper', data['moving'] & (state[:, 1] > 2.5), [2.5,3.5]),
             ('moving lower', data['moving'] & (state[:, 1] < 2.), [4.5,1.5])]
  fig, axes = plt.subplots(len(targets), 2, figsize=(12, 13), constrained_layout=True)
  records = []
  for row, (kind, mask, target) in enumerate(targets):
    pool = np.flatnonzero(mask)
    index = pool[np.argmin(np.linalg.norm(state[pool, :2]-target, axis=-1))]
    s = state[index]
    xs, ys = np.linspace(s[0]-.6, s[0]+.6, 51), np.linspace(s[1]-.6, s[1]+.6, 51)
    xx, yy = np.meshgrid(xs, ys)
    xy = np.stack([xx.ravel(), yy.ravel()], axis=-1)
    states = jnp.repeat(jnp.asarray(s)[None], len(xy), axis=0)
    candidates = shift_with_newest(states, jnp.asarray(xy)[:, None])
    set_values = np.asarray(nearest_reference(candidates, references, mean, std)['distance2'])[:, 0]
    mmd_values = np.asarray(mmd_components(candidates, references, mean, std, bandwidths)['mmd'])
    persistence = shift_with_newest(jnp.asarray(s)[None], jnp.asarray(s[:2])[None,None])
    nearest = int(nearest_reference(persistence, references, mean, std)['index'][0,0])
    endpoint = geometry(np.asarray(states), np.asarray(candidates))['endpoint_invalid_fraction'] == 0
    for col, (title, values, grad) in enumerate([
        ('MMD of a point-mass candidate', mmd_values, data['persistence_mmd_gradient'][index]),
        ('Full-F4 hard nearest-set cost', set_values, data['persistence_set_gradient'][index])]):
      axis = axes[row, col]
      z = np.ma.array(values.reshape(xx.shape), mask=~endpoint.reshape(xx.shape))
      contour = axis.contourf(xx, yy, z, levels=16, cmap='viridis')
      fig.colorbar(contour, ax=axis, shrink=.7)
      axis.scatter(data['references'][:,0], data['references'][:,1], s=8, c='white', alpha=.45)
      axis.scatter(data['control'][index,:,0], data['control'][index,:,1], s=15, color='C0', label='Diagonal samples')
      axis.scatter(data['primary'][index,:,0], data['primary'][index,:,1], s=15, color='C1', label='MMD samples')
      axis.plot(s.reshape(4,2)[:,0], s.reshape(4,2)[:,1], 'kx-', label='Input history')
      descent = -.2*grad/(np.linalg.norm(grad)+1e-12)
      axis.arrow(s[0],s[1],descent[0],descent[1],width=.005,color='red',length_includes_head=True)
      axis.set(xlim=(xs[0],xs[-1]), ylim=(ys[0],ys[-1]), aspect='equal',
               title=f'{kind}, ep {data["episode_id"][index]}, t {data["timestep"][index]}\n{title}')
      if row == col == 0:
        axis.legend(fontsize=7)
    records.append({'context_index': int(index), 'kind': kind, 'episode_id': int(data['episode_id'][index]),
                    'timestep': int(data['timestep'][index]), 'state': s, 'persistence_nearest_reference_index': nearest,
                    'nearest_reference': np.asarray(references[nearest]),
                    'persistence_descent_direction_mmd': -data['persistence_mmd_gradient'][index],
                    'persistence_descent_direction_set': -data['persistence_set_gradient'][index]})
  fig.savefig(out/'objective_landscapes.png', dpi=150)
  plt.close(fig)
  write_json(out/'landscape_examples.json', {'selection': 'nearest saved observable states to predeclared XY targets within each group',
             'landscape': 'delta-law candidate at newest XY, exact old-frame shift; red arrows are descent DIRECTION only, rescaled to length 0.2',
             'geometry': 'blocked endpoints masked; contours do not enforce the anchor bound or certify reachability', 'examples': records})


def plot_switch(out, data, switches):
  cases = switches['first_changed_examples_in_saved_context_order']
  if not cases:
    return
  case = cases[0]
  index = case['context_index']
  direction_index = next(i for i,n in enumerate(case['references_after']) if n != case['reference_before'])
  direction = np.asarray(case['directions'][direction_index])
  s = data['state'][index]
  t = np.linspace(-.05,.05,201)
  xy = s[:2] + t[:,None]*direction
  states = np.repeat(s[None],len(t),axis=0)
  y = shift_with_newest(jnp.asarray(states),jnp.asarray(xy)[:,None])
  detail = nearest_reference(y,jnp.asarray(data['references']),jnp.asarray(data['mean']),jnp.asarray(data['std']))
  indices = np.asarray(detail['index'])[:,0]
  cost = np.asarray(detail['distance2'])[:,0]
  derivative = (2*(xy-data['references'][indices,:2])/data['std'][:2]**2*direction).sum(-1)
  fig, axes = plt.subplots(1,3,figsize=(13,3.5),constrained_layout=True)
  for axis, value, title in zip(axes,(cost,indices,derivative),('Continuous set cost','Selected reference index','Directional derivative')):
    axis.plot(t,value)
    axis.set(xlabel='Signed newest-XY perturbation (maze units)',title=title)
  fig.suptitle(f'First 0.01 reference-switch example: ep {data["episode_id"][index]}, t {data["timestep"][index]}')
  fig.savefig(out/'nearest_reference_switch.png',dpi=170)
  plt.close(fig)


def write_report(out, config, sampled, paired, controlled, trajectory, switches, collapse):
  original = read(Path(config['arguments']['source_run'])/'config.json')
  first = sampled[CONTROL]['16']['mean_over_sampling_seeds']
  second = sampled[PRIMARY]['16']['mean_over_sampling_seeds']
  change = paired[PRIMARY]['16']
  candidates = controlled['candidates']
  frozen = lambda name: candidates[name]['frozen_nonreset']
  formula = read(out/'formula_checks.json')
  gradients = read(out/'gradients.json')
  replay = read(out/'baseline_reproduction.json')
  root = config['arguments']['source_run']
  lines = [
      '# PointMaze failure-objective diagnostic', '',
      '**Decision B: reconsider the unconditional one-step target before another training comparison.** '
      'Hard set distance rejects the most conspicuous MMD dispersion artifact, but it still rewards moving every '
      'held-out frozen history to another stored location and can reward complete sample collapse. '
      'Neither objective passes the stronger behavioral test of preserving an already stationary outcome without '
      'an independent reason to move it. This is structural/observational evidence, not proof that these histories '
      'are absorbing deaths. No new neural network was trained and no A/B/C comparison was launched.', '',
      '## Preserved context and compatibility', '',
      f'The authoritative run remains `{root}`. Its README designation, sibling run records and Git history were '
      f'inspected. Remote branch HEAD was `{config["authority"]["verified_remote_head"]}`, matching local HEAD; '
      'no newer authoritative run was found. All five guarded final checkpoints were available and their SHA-256 '
      'hashes match the saved metrics. The original K=8/16 seed-0 MMD results and separate K=64 frozen-history '
      'stationarity check reproduce. Every original artifact hash is unchanged after the diagnostic.', '',
      f'- Environment/data: `point_two_route_swamp_windy_f4_v0`, per-cell swamp probability 0.30; dataset SHA-256 '
      f'`{original["dataset"]["dataset"]["sha256"]}`.',
      '- Expert-positive source episodes [1200,6000), including failures and stationary rows; 4,321 training '
      'episodes/216,050 transitions and 479 validation episodes/23,950 transitions. The source is teacher-generated '
      'with documented random behavior, not uniformly optimal or success-only data.',
      '- Reuse all 1,024 saved evaluation contexts from 427 episodes; frozen non-reset subset: 91 contexts from '
      '39 episodes; moving histories: 863 contexts. The lower-corridor moving stratum has only 7 contexts/5 episodes.',
      '- Reuse exactly 233 training-only F4 references: 141 uniform-coverage, 12 expert-positive and 80 appended '
      'blind-demonstrator source episodes. Zero source-validation overlap. The original bank composition used '
      'privileged teacher-mode/swamp bits; this diagnostic does not construct a bank or use hidden labels. '
      'Reference-set semantics do not remove that privileged provenance.',
      f'- Bank content SHA-256: `{config["references"]["bank_content_sha256"]}`. Saved episode IDs and rows are '
      'checked against the source observations. State is newest-first F4 XY, width 8; action is two coordinates '
      'in [-1,1]; the width-8 commanded goal stays in both frozen policies and transition inputs.',
      f'- Frozen nominal: `{original["arguments"]["nominal"]}/best.pkl`; frozen actor: '
      f'`{original["arguments"]["actor"]}`. Models: guarded seed 0/1 diagonal-only and L=0.25 MMD-trained, '
      'plus seed-0 L=0.75 MMD-trained. The actor originally saw all source episodes, so this is not held-out actor evaluation.', '',
      '`config.json` records exact input/checkpoint hashes, split/reference IDs, normalizers and bandwidths. '
      'Normalization is loaded unchanged and independently reproduced from training data. No new labels, '
      'critic score, death reward, hidden input, learned embedding, or normalization fit was introduced.', '',
      '## Objectives and estimators', '',
      '`L_set(Y) = mean_i min_f sum_j ((Y_ij - F_fj) / std_j)^2` uses all eight F4 coordinates. '
      'Both sides use the same original normalization; its common mean cancels in Euclidean differences. '
      'The primary cost is a hard minimum. Exact ties select the first reference in saved order; the continuous '
      'piecewise-quadratic value has a potentially discontinuous selected index/gradient at a tie. '
      'The implementation differentiates that chosen branch, without a soft minimum or tie-gradient averaging.', '',
      '`MMD_V = GG + RR - 2 GR` reuses the original full-F4 multiscale Gaussian kernel, with bandwidths '
      f'`{config["kernel"]["bandwidths"]}`. Each objective is computed per fixed (s,x) group. '
      'The actor action is held fixed across models, K, and repetitions; each sample first draws x_prime from '
      'the frozen nominal model and then a transition. Four paired sampling seeds (0,1,2,3) and K=8/16 '
      'are evaluated on every checkpoint. Contexts are never pooled before calculating MMD.', '',
      'Because generated self pairs have kernel value 1, `GG_U = (K*GG_V - 1)/(K-1)` removes their bias. '
      '`MMD_generated_U = GG_U + RR - 2 GR` is unbiased over independent generated draws against the fixed '
      'empirical bank. RR retains all pairs, including bank self pairs; it is an exact expectation under that '
      'empirical target. This is not a population-failure estimator, and a realized U estimate may be negative.', '',
      '## Paired decomposition: where the MMD gain comes from', '',
      f'For seed 0, L=0.25, K=16, averaging all four sampling seeds: GG '
      f'{first["all"]["gg"]:.6f} -> {second["all"]["gg"]:.6f}; RR stays '
      f'{first["all"]["rr"]:.6f}; GR {first["all"]["gr"]:.6f} -> {second["all"]["gr"]:.6f}. '
      f'Total MMD {first["all"]["mmd"]:.6f} -> {second["all"]["mmd"]:.6f}; '
      f'nearest-set cost {first["all"]["set_cost"]:.6f} -> {second["all"]["set_cost"]:.6f}.', '',
      f'The MMD change is `{change["all"]["gg"]["mean"]:.6f} + 0 + '
      f'{change["all"]["cross_contribution"]["mean"]:.6f} = {change["all"]["mmd"]["mean"]:.6f}`. '
      'All improvement comes from reduced generated-generated similarity; reduced cross-reference similarity '
      'actually offsets part of it. This conclusion uses the kernel terms directly, not a visual inference from diversity.', '',
      f'On the 91 frozen histories, the corresponding changes are GG '
      f'{change["frozen_nonreset"]["gg"]["mean"]:.6f}, cross contribution '
      f'{change["frozen_nonreset"]["cross_contribution"]["mean"]:.6f}, total '
      f'{change["frozen_nonreset"]["mmd"]["mean"]:.6f}, and set cost '
      f'{change["frozen_nonreset"]["set_cost"]["mean"]:+.6f}. '
      'When the old frames are fixed, within-group GG depends only on newest-XY spread. For histories far from '
      'the bank, cross-kernel similarity can be weak while dispersion still lowers GG. This explains the surrogate mismatch.', '',
  ]
  for name in paired:
    values=paired[name]['16']['all']
    lines.append(f'- `{name}` versus its seed-matched control: MMD change {values["mmd"]["mean"]:.6f} '
                 f'(95% episode interval {np.round(values["mmd"]["ci95_episode"],6).tolist()}); '
                 f'set-cost change {values["set_cost"]["mean"]:+.6f}.')
  lines += ['', 'Intervals resample source episodes after averaging sampling replicates per context, retaining '
            'context weights through episode sums/counts. Repeated frozen rows are not treated as independent episodes. '
            'The fixed models/bank and prior reuse of this development subset are not covered by those intervals.', '',
            '![MMD term decomposition](mmd_decomposition.png)', '',
            f'For the seed-0 control, K8/K16 V estimates are '
            f'{sampled[CONTROL]["8"]["mean_over_sampling_seeds"]["all"]["mmd"]:.6f}/'
            f'{first["all"]["mmd"]:.6f}, while generated-U estimates are '
            f'{sampled[CONTROL]["8"]["mean_over_sampling_seeds"]["all"]["mmd_generated_u"]:.6f}/'
            f'{first["all"]["mmd_generated_u"]:.6f}. The analogous trained estimates show the same estimator effect. '
            'A lower V estimate at K16 is not a better model. Residual K differences are Monte Carlo error; K8/16 '
            'draws share keys but are not nested sample sets.', '', '![K and self-pair bias](sample_count_bias.png)', '',
            '## Frozen histories and controlled candidates', '',
            'The saved K64 check reproduces stationary sampling rates of '
            f'{replay["frozen_stationary_fraction_K64"][CONTROL]:.2%} (control) and '
            f'{replay["frozen_stationary_fraction_K64"][PRIMARY]:.2%} (MMD-trained). '
            'Their recorded next positions are stationary, but observational stationarity is not a proof of absorbing death.', '',
            'For controlled candidate comparisons below, K=16 and sampling seed 0 are fixed. Persistence is '
            '`[current XY, state[0:6]]`; small cardinal perturbations and symmetric radius-0.05 jitter change only '
            'newest XY and use the existing endpoint projection. For moving histories persistence is merely a '
            'valid candidate, not a failure label.', '',
            f'On frozen contexts, persistence/diagonal-only/MMD-trained set costs are '
            f'{frozen("persistence")["set_cost"]:.6f}/{frozen("diagonal_only")["set_cost"]:.6f}/'
            f'{frozen("mmd_trained")["set_cost"]:.6f}; MMD values are '
            f'{frozen("persistence")["mmd"]:.6f}/{frozen("diagonal_only")["mmd"]:.6f}/'
            f'{frozen("mmd_trained")["mmd"]:.6f}. MMD prefers its trained output over persistence in all 91 contexts; '
            'hard set cost prefers persistence in all 91. Symmetric 0.05 jitter lowers MMD in all 91, '
            f'but raises mean set cost to {frozen("symmetric_jitter_radius_0.05")["set_cost"]:.6f}. '
            'Thus set distance is better aligned with this specific stationary-candidate comparison.', '',
            'However, no held-out persistence vector is an exact bank member. Moving newest XY to the reference '
            'that minimizes old-frame distance reduces set cost for every frozen context: '
            f'{frozen("persistence")["set_cost"]:.6f} -> {frozen("history_nearest_collapse_raw")["set_cost"]:.6f}, '
            f'with mean movement {frozen("history_nearest_collapse_raw")["motion"]:.4f} maze units. '
            'All these frozen-context moves pass endpoint, per-coordinate cap and segment checks, but '
            f'{frozen("history_nearest_collapse_raw")["anchor_bound_violation_fraction"]:.2%} of saved draw/radius '
            'pairs fail the original action anchor bound. Projection does not remove the frozen-context drift.', '',
            f'Even restricting to contexts where a single collapsed position satisfies all 16 sampled anchors and '
            f'the geometry checks leaves {collapse["groups"]["frozen_nonreset"]["all_sampled_anchors_and_geometry_pass_contexts"]} '
            'frozen contexts. Their exact counts and costs are in `feasible_collapse_probe.json`. This empirical '
            'subset is not an all-randomness guarantee. The formula contains no penalty for losing conditional '
            'uncertainty. Without counterfactual ground truth, we cannot determine whether the removed spread '
            'is legitimate uncertainty or model error; this demonstrates a collapse loophole, not that the '
            'original spread should be retained.', '',
            '![Controlled candidate objectives](controlled_candidates.png)', '',
            '## History mismatch, local derivatives, and feasibility', '',
            f'For frozen persistence, fixed old frames contribute '
            f'{frozen("persistence")["set_history"]/frozen("persistence")["set_cost"]:.2%} of nearest-set cost. '
            f'The immutable history-only floor averages {frozen("persistence")["history_floor"]:.6f}; across all '
            f'contexts it is {candidates["persistence"]["all"]["history_floor"]:.6f}. '
            'Distance to each fixed reference has a fixed old-frame contribution, but the chosen reference may '
            'change when newest XY changes; consequently the selected old-frame term can change without editing history.', '',
            f'A 0.01 maze-unit cardinal perturbation changes the nearest reference in at least one direction for '
            f'{switches["scales"]["0.01"]["frozen_nonreset"]["any_direction_switch_fraction"]:.2%} of frozen contexts '
            f'and {switches["scales"]["0.01"]["all"]["any_direction_switch_fraction"]:.2%} overall. '
            'The cost remains continuous while the selected-reference derivative can jump. Reference indices and '
            'their exact source episode/row IDs are saved in `primary_nearest_references.json`.', '',
            '![Nearest-reference switch](nearest_reference_switch.png)', '',
            f'All newest-XY gradients are finite. At frozen persistence, mean common-translation gradient norms '
            f'are {gradients["persistence"]["mmd"]["frozen_nonreset"]["translation_gradient_norm"]:.6f} for MMD '
            f'and {gradients["persistence"]["set"]["frozen_nonreset"]["translation_gradient_norm"]:.6f} for set cost. '
            'Their numerical scales have different units and are not comparable quality scores. Neither objective '
            'has zero local drift on these held-out frozen points. Old frames stay fixed in these derivatives.', '',
            'The saved actual own-diagonal-anchor-to-generated displacement is also evaluated at fractions '
            '0, 0.25, 0.5, 0.75 and 1. This is not a pairing of separately trained control and treatment samples. '
            'Along the actual residual path, MMD falls while set cost rises on average for frozen histories. '
            'Interpolation does not reproject the path; every fraction has separate endpoint/segment/bound checks.', '',
            '![Loss along actual displacement](anchor_displacement_losses.png)', '',
            '![Local objective landscapes](objective_landscapes.png)', '',
            f'The unrestricted history-nearest collapse violates the one-step cap in '
            f'{candidates["history_nearest_collapse_raw"]["all"]["one_step_cap_violation_fraction"]:.2%} of all '
            f'contexts. For all {config["groups"]["moving_lower_corridor"]["contexts"]} lower-corridor moving '
            'contexts, the reference lies across blocked geometry and beyond the coordinate cap. Their large '
            'set costs distinguish public spatial/history configurations, not proven failure versus safety. '
            'This objective can favor unrelated failure locations; it is not merely motion suppression. '
            'Endpoint and sampled-segment checks do not establish full physical reachability. All evaluated model '
            'outputs retain exact history shift, valid endpoints and their sampled anchor bounds.', '',
            '## Formula properties versus behavioral evidence', '',
            f'On the 233 exact training-bank vectors, set cost is zero as expected. Repeating one bank point '
            f'16 times gives set cost {formula["repeated_single_reference_set_cost"]:.1f} but MMD '
            f'{formula["repeated_single_reference_mmd"]:.6f}. Duplicating that reference 233 times in memory leaves '
            f'set cost unchanged and changes MMD to {formula["reweighted_reference_mmd"]:.6f}. The bank file is '
            'unchanged. These checks establish membership, duplication invariance and frequency sensitivity; '
            'they do not establish failure recognition. The held-out behavior results above are separate.', '',
            '## Recommendation and scope of the evidence', '',
            'Set distance is a more appropriate diagnostic for proximity to a reference set than unconditional '
            'distribution matching, and it exposes the MMD artifact. Nevertheless, nearest-point drift, '
            'uncertainty collapse, fixed-history floors and unreachable references remain basic obstacles to '
            'treating it as a reasonable global single-step failure objective. Reconsider reference conditioning '
            'and one-step versus longer-horizon target compatibility before another training experiment. This '
            'diagnostic does not establish that the F4 representation itself is insufficient.', '',
            'No hyperparameter of the alternative was fitted or tuned: it has the prescribed hard minimum, '
            'unchanged normalization and bank. This repeatedly used development subset is not an independent '
            'validation of a selected objective. No hidden labels, outcome classifier or generated-state failure '
            'labels were used. Counterfactual ground truth remains unavailable in these records.', '',
            '## Completed reproduction and next experiment', '',
            f'Completed: seven formula/gradient tests, a 128-context smoke diagnostic, and the full '
            f'5-checkpoint x 2-K x 4-sampling-seed diagnostic in {config["elapsed_seconds"]:.1f} seconds on '
            f'{config["runtime"]["devices"]}. The report reads saved samples; it does not resample a model.', '',
            '```bash', 'python -m scripts.test_failure_objectives',
            'python -m ett.diagnose_failure_objectives --out-dir artifacts/ett_objective_diagnostic/smoke --max-contexts 128 --sampling-seeds 0 --sample-counts 16 --verified-remote-head b44562f1b32d45b1b42f5b54bbbbfc1ae1d39a39',
            config['command'], f'python -m ett.report_failure_objectives --run-dir {out.as_posix()}', '```', '',
            'Use a fresh output directory when reproducing; previous artifacts are never overwritten by the '
            'diagnostic. `visualization_samples.npz` contains only visible evaluation states, sampled outcomes '
            'and objective gradients, not weights, hidden labels or training targets. Publication of the code, '
            'configuration, metrics and evaluation artifacts was separately authorized by the user. '
            'Checkpoints remain local and are excluded from publication.', '',
            'The concrete later A/B/C protocol is in [NEXT_ABC_SPEC.md](NEXT_ABC_SPEC.md). In that specification '
            'A is a plain conditional mixture with diagonal fitting only; B is the identical plain network with '
            'diagonal plus hard-set cost and no equality indicator or zero-residual architecture; C uses the '
            'current explicitly anchored architecture with the same two losses. The selected hard-set cost is '
            'an exploratory candidate, not approved as a failure metric. No A/B/C implementation, training run '
            'or launch command was executed.']
  (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
  (out/'NEXT_ABC_SPEC.md').write_text('''# Later A/B/C experiment specification (NOT RUN)

This is a bounded exploratory protocol for a separately authorized follow-up.
The present decision is to reconsider conditional references and target-horizon
compatibility first; the current diagnostic does not validate either objective.
If an A/B/C comparison is nevertheless authorized, preregister this protocol
without repeatedly tuning it on the 91 frozen histories.

1. Freeze the same nominal policy and actor; retain the same expert-source
   episode split, commanded goals, 233 training-only references and fixed
   normalization. Use the original diagonal initialization for every arm.
   No hidden input, new bank, death reward or critic objective is introduced.
2. A: use a plain 128/128 conditional zero-inflated three-Gaussian transition
   network with input [s,x,x_prime,g], only newest XY stochastic, and train only
   the existing mixed-measure diagonal NLL. Off-diagonal sampling evaluates
   these ordinary concatenated inputs. There is no equality flag or residual
   multiplied by action difference.
3. B: exactly the same plain network and initialization as A; train diagonal
   NLL plus lambda times the hard full-F4 nearest-set cost. There is no equality
   flag, explicit zero-on-equality residual, or enforced diagonal identity.
   The network must learn empirical diagonal preservation from NLL alone.
4. C: the current guarded architecture, initialized from the same diagonal
   model, with its zero-initialized 64/64 residual. Train the same NLL plus the
   same lambda/set cost as B. Retain L=0.25 in raw XY/action Euclidean units.
   C forces equality with its OWN CURRENT diagonal sampling law at x=x_prime;
   because its base parameters are updated, it does not freeze the initial
   observational distribution. This distinction must be tested empirically.
5. All arms retain exact history shifts and the existing endpoint/cap geometry.
   A and B have no guaranteed anchored action-change bound: measure violations
   against their own shared-randomness diagonal draws rather than silently
   projecting them into C's feasible set. C enforces its bound with the guard
   and fallback. Endpoint/segment checks are not a dynamics proof.
6. Fixed budget: seed 0 and 1, 500 updates each, batch 64, K=8 for pessimistic
   training; K=8/16 for evaluation. Use Adam at 1e-5 for all trainable parameters
   in all arms, without weight decay or an extra soft constraint. Choose one
   common lambda for B/C using four TRAINING batches (seeds 300..303): match a
   weighted pessimistic/diagonal gradient norm ratio of 0.01 using the median
   over both architectures, clipped to [1e-4,1]. Record the actual objective and
   gradient scales before training; do not select lambda on evaluation outcomes.
   Keep the final fixed-budget checkpoint primary. Run finite-gradient and
   parameter-change checks first; record discrete-gate gradient limitations
   consistently in B/C rather than changing estimators between arms.
7. Measure empirical diagonal retention against initialization AND A with
   episode-paired NLL changes, mixed atom calibration, position/energy errors,
   and visible frozen/moving/region strata. Report uncertainty, not just an
   overall average dominated by common near-goal rows. A provisional engineering
   guardrail is no more than 0.05 nats extra NLL deterioration versus A; it is
   not a theorem or a causal criterion. Report off-diagonal set/MMD terms,
   stationarity drift, diversity/collapse and geometry/bound diagnostics even
   if the diagonal guardrail passes.
8. Treat the current saved evaluation subset as development cases. Use newly
   held-out expert-source episodes for a later independent diagonal assessment;
   obtaining them is separate future work. Do not claim counterfactual accuracy
   without a verified simulator state-restoration/randomness protocol or valid
   off-diagonal outcome data. Observed stationarity alone is not an absorbing
   death label.

A versus B measures how this objective affects a plain network's empirical
diagonal fit and sampled outcomes. B versus C measures the combined influence
of anchoring, the hard action bound and added residual capacity. It does NOT
isolate an equality label, prove that structure alone preserves a pretrained
diagonal density, or separate these architectural factors causally. Different
optimization difficulty and C's extra parameters must be reported. None of
the arms can establish the true counterfactual kernel, a worst-case Q function,
or genuine failure recognition merely by decreasing the objective.

No A/B/C model code, optimizer run, or command has been launched in this task.
''',encoding='utf-8')


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--run-dir',required=True)
  args=parser.parse_args()
  out=Path(args.run_dir)
  config=read(out/'config.json')
  if config['status']!='complete' or config['arguments']['max_contexts'] is not None:
    raise ValueError('report requires the completed full saved-context diagnostic')
  data=dict(np.load(out/'visualization_samples.npz',allow_pickle=False))
  sampled=read(out/'sampled_objectives.json')
  paired=read(out/'paired_changes.json')
  controlled=read(out/'controlled_candidates.json')
  trajectory=read(out/'anchor_to_generated.json')
  switches=read(out/'reference_switches.json')
  collapse=collapse_probe(data)
  write_json(out/'feasible_collapse_probe.json',collapse)
  plot_decomposition(out,paired)
  plot_k(out,sampled)
  plot_candidates(out,controlled)
  plot_interpolation(out,trajectory)
  plot_landscapes(out,data)
  plot_switch(out,data,switches)
  write_report(out,config,sampled,paired,controlled,trajectory,switches,collapse)
  write_json(out/'report_provenance.json',{
      'completed_command':f'python -m ett.report_failure_objectives --run-dir {args.run_dir}',
      'source_sha256':file_sha('ett/report_failure_objectives.py'),
      'no_new_model_draws_or_updates':True})
  print(f'wrote report and six figures to {out}',flush=True)


if __name__=='__main__':
  main()
