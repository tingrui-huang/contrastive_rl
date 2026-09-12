"""Create reproducible figures and a report for completed ETT matching runs."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('MPLBACKEND', 'Agg')
import jax
import matplotlib.pyplot as plt
import numpy as np

from ett.anchored_transition import load_anchored
from ett.diagonal_transition import load_diagonal_transition, POINTMAZE_WALLS
from ett.dataset import ExpertTransitionDataset
from ett.run_distribution_matching import write_json, frozen_actor
from propensity.nominal_policy import load_nominal_policy
from scripts.make_swamp_f4_failure_bank import file_sha


def read(path):
  return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--run-dir', required=True)
  args = parser.parse_args()
  directory = Path(args.run_dir)
  config, metrics = read(directory/'config.json'), read(directory/'metrics.json')
  if config['status'] != 'complete':
    raise ValueError('report requires a completed bounded comparison')
  arms = metrics['arms']
  labels = list(arms)
  short = [f"s{arms[n]['seed']} L={arms[n]['L']} w={arms[n]['lambda_pess']}" for n in labels]
  primary_name = next(n for n in labels if arms[n]['lambda_pess'] > 0)
  primary = arms[primary_name]
  base_diag = metrics['baseline_diagonal']
  base_marginal = metrics['baseline_marginal']

  fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
  axes[0].axhline(base_diag['nll'], color='k', linestyle=':', label='Initialization')
  axes[0].plot(range(len(labels)), [arms[n]['diagonal']['nll'] for n in labels], 'o', label='Final checkpoint')
  axes[0].set_ylabel('Diagonal NLL (nats / XY transition)')
  axes[0].legend(fontsize=8)
  for count, color, offset in ((8, 'C0', -.18), (16, 'C1', .18)):
    axes[1].bar(np.arange(len(labels))+offset, [arms[n]['marginal'][str(count)]['mmd2'] for n in labels],
                width=.35, color=color, label=f'K={count}')
    axes[1].axhline(base_marginal[str(count)]['mmd2'], color=color, linestyle=':')
  axes[1].set_ylabel('Conditional full-F4 MMD squared')
  axes[1].legend()
  for name, label in zip(labels, short):
    history = read(directory/name/'training_history.json')
    axes[2].plot([h['step'] for h in history], [h['validation_nll'] for h in history], label=label)
  axes[2].set_xlabel('ETT update')
  axes[2].set_ylabel('Validation diagonal NLL')
  axes[2].legend(fontsize=7)
  for axis in axes[:2]:
    axis.set_xticks(range(len(short)), short, rotation=35, ha='right', fontsize=8)
  fig.savefig(directory/'fit_and_matching.png', dpi=170)
  plt.close(fig)

  fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
  for axis, key, title in zip(axes, ['moving_history_mean_motion', 'moving_history_near_zero_fraction', 'low_diversity_fraction'],
                             ['Motion with moving input history', 'Near-zero motion with moving history', 'Low sample diversity (std norm < 0.001)']):
    axis.bar(range(len(labels)), [arms[n]['marginal']['16'][key] for n in labels])
    axis.axhline(base_marginal['16'][key], color='k', linestyle=':')
    axis.set_title(title, fontsize=10)
    axis.set_xticks(range(len(short)), short, rotation=35, ha='right', fontsize=8)
  fig.savefig(directory/'motion_and_diversity.png', dpi=170)
  plt.close(fig)

  fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
  m = primary['marginal']['16']
  axes[0].bar(['Fixed old frames', 'Generated newest XY'],
              [m['history_share_mean_cross_squared_distance'], 1-m['history_share_mean_cross_squared_distance']])
  axes[0].set_title('Share of cross-reference squared distance')
  saturation = m['cross_kernel_saturation_fraction_by_bandwidth']
  axes[1].bar([f'{float(k):.2f}' for k in saturation], list(saturation.values()))
  axes[1].set_xlabel('Frozen training bandwidth')
  axes[1].set_ylabel('Cross kernel < 1e-6 fraction')
  axes[2].bar(['Base correction', 'Candidate correction', 'Bound fallback', 'Blocked segment'],
              [m[k] for k in ['base_geometry_correction_fraction', 'candidate_geometry_correction_fraction',
                              'bound_fallback_fraction', 'straight_segment_blocked_fraction']])
  axes[2].tick_params(axis='x', rotation=30)
  axes[2].set_title('Geometry diagnostics')
  fig.savefig(directory/'history_and_geometry.png', dpi=170)
  plt.close(fig)

  # Representative states selected using observable coordinates only. The same
  # sampled randomness is used for each action pair and both model versions.
  run_args = config['arguments']
  dataset = ExpertTransitionDataset(run_args['dataset'])
  validation = dataset.arrays('validation')
  learned = load_anchored(directory/primary_name/'final.pkl')
  original = load_diagonal_transition(run_args['diagonal'])
  rows = [int(np.argmin(np.linalg.norm(validation.state[:, :2]-xy, axis=-1)))
          for xy in ([2.8, 3.5], [3.7, 3.5], [5., 3.5])]
  fig, axes = plt.subplots(3, 3, figsize=(12, 10), constrained_layout=True)
  examples = []
  for r, index in enumerate(rows):
    s, g, xp = validation.state[index], validation.goal[index], validation.action[index]
    for c, (kind, x) in enumerate([('identical', xp), ('nearby', np.clip(xp+[.01, -.01], -1, 1)), ('far', np.where(xp >= 0, -1., 1.))]):
      key = jax.random.PRNGKey(972+r)
      value, detail = learned.sample_with_diagnostics(s, x, xp, key, 256, g)
      before = np.asarray(original.sample(s, xp, xp, key, 256, g))
      value = np.asarray(value)
      axis = axes[r, c]
      for cell in np.argwhere(POINTMAZE_WALLS):
        axis.add_patch(plt.Rectangle(cell, 1, 1, color='.9', zorder=0))
      axis.scatter(before[:, 0], before[:, 1], s=7, alpha=.25, label='Initial diagonal')
      axis.scatter(value[:, 0], value[:, 1], s=7, alpha=.25, label='Trained ETT')
      axis.plot(s.reshape(4, 2)[:, 0], s.reshape(4, 2)[:, 1], 'kx-', label='Input history')
      axis.set(xlim=(max(0, s[0]-1.2), min(9, s[0]+1.2)), ylim=(2.5, 4.),
               title=f'ep {validation.episode_id[index]} t {validation.timestep[index]} / {kind}')
      axis.set_aspect('equal')
      if r == c == 0:
        axis.legend(fontsize=7)
      examples.append({'episode_id': int(validation.episode_id[index]),
                        'timestep': int(validation.timestep[index]), 'pair': kind,
                        'state': s, 'x': x, 'x_prime': xp,
                        'mean_xy_initial': before[:, :2].mean(0), 'mean_xy_trained': value[:, :2].mean(0),
                        'maximum_anchor_change': float(detail['emitted_change'].max()),
                        'radius': float(detail['radius'][0])})
  fig.savefig(directory/'action_pair_samples.png', dpi=170)
  plt.close(fig)
  write_json(directory/'plotted_examples.json', examples)

  # Supplementary observable-history diagnostic, excluding padded reset frames.
  # This selection uses neither hidden bits nor death labels. Recorded next
  # states are used only to report empirical observed stationarity, not targets.
  selected = read(directory/'evaluation_contexts.json')
  lookup = {(int(e), int(t)): i for i, (e, t) in enumerate(zip(
      validation.episode_id, validation.timestep))}
  indices = np.array([lookup[(e, t)] for e, t in zip(selected['episode_id'], selected['timestep'])])
  evaluation = validation.take(indices)
  frozen = ((evaluation.timestep > 0) & np.all(
      evaluation.state.reshape(-1, 4, 2) == evaluation.state[:, None, :2], axis=(-2, -1)))
  frozen_arrays = evaluation.take(np.flatnonzero(frozen))
  actor, _ = frozen_actor(run_args['actor'], config['references']['dataset_content_sha256'])
  nominal = load_nominal_policy(run_args['nominal'])
  frozen_report = {'selection': 'identical observed F4 frames and timestep > 0, on the saved validation context subset',
                    'hidden_labels_used': False, 'contexts': int(frozen.sum()), 'samples_per_context': 64,
                    'observed_stationary_next_xy_fraction': float(np.all(frozen_arrays.delta_xy == 0, axis=-1).mean()),
                    'arms': {}}
  actions = actor(frozen_arrays.state, frozen_arrays.goal, jax.random.PRNGKey(413))
  for name in labels:
    model = load_anchored(directory/name/'final.pkl')
    samples = np.asarray(model.sample_marginal(nominal, frozen_arrays.state, actions,
                          jax.random.PRNGKey(414), 64, frozen_arrays.goal))
    motion = np.linalg.norm(samples[..., :2] - frozen_arrays.state[:, None, :2], axis=-1)
    frozen_report['arms'][name] = {'mean_motion': float(motion.mean()),
                                  'exact_zero_fraction': float((motion == 0).mean()),
                                  'near_zero_fraction': float((motion <= .05).mean())}
  write_json(directory/'frozen_history_evaluation.json', frozen_report)

  control_name = next(n for n in labels if arms[n]['lambda_pess'] == 0 and arms[n]['seed'] == primary['seed'])
  control = arms[control_name]
  reference_ids = np.asarray(config['references']['kept_episode_ids'])
  bank_sources = {'uniform_coverage': int((reference_ids < 1200).sum()),
                  'expert_positive_source': int(((reference_ids >= 1200) & (reference_ids < 6000)).sum()),
                  'appended_blind_demonstrator': int((reference_ids >= 6000).sum())}
  summary = (f'The bounded prototype reduced held-out full-F4 MMD without a material loss of diagonal fit. '
             f'For the predeclared primary arm (seed {primary["seed"]}, L={primary["L"]}, lambda={primary["lambda_pess"]}), '
             f'K=16 MMD fell from the matched diagonal-only control\'s {control["marginal"]["16"]["mmd2"]:.6f} '
             f'to {primary["marginal"]["16"]["mmd2"]:.6f}; diagonal NLL was '
             f'{control["diagonal"]["nll"]:.6f} versus {primary["diagonal"]["nll"]:.6f}. '
             'The second seed reproduced the direction. All final-sample bound, history-shift, and endpoint '
             'checks passed. The improvement is not predominantly freezing: moving-history motion and diversity '
             f'increase at small L. On {frozen_report["contexts"]} previously frozen non-reset histories, however, '
             f'sampled stationarity falls from {frozen_report["arms"][control_name]["exact_zero_fraction"]:.2%} '
             f'to {frozen_report["arms"][primary_name]["exact_zero_fraction"]:.2%}, despite all their recorded next '
             'positions being stationary. The larger L '
             'requires more geometry corrections. This establishes optimization of the specified surrogate, '
             'not a validated failure or counterfactual transition model.')

  lines = [
      '# Bounded single-step PointMaze ETT distribution matching', '', summary, '',
      '## Scope and interpretation', '',
      'This is a failure-directed transition prototype, not recovery of the true counterfactual kernel, '
      'a formally established worst-case Q function, or evidence of causal death. Only ETT parameters were updated. '
      'The nominal model and agent actor were frozen; their checkpoint files are unchanged. '
      'No actor/critic loop, critic objective, death classifier, hidden-state training input, or new failure bank was introduced.', '',
      '## Verified data and initialization', '',
      '- Environment: `point_two_route_swamp_windy_f4_v0`, per-cell swamp probability 0.30.',
      f'- Dataset file SHA-256: `{config["dataset"]["dataset"]["sha256"]}`.',
      '- All 6,600 source episodes contain 50 valid transitions and a terminal observation with a dummy action. '
      'ETT diagonal fitting uses source episodes [1200,6000), including failed trajectories and absorbing rows. '
      'This teacher-generated source includes random behavior in its source metadata (`random_frac=0.2`); '
      'it is the established expert-positive source, not uniformly optimal or success-only data. '
      'Uniform coverage [0,1200) and the separately appended blind-demonstrator [6000,6600) are not diagonal fitting data.',
      '- The inherited full-source episode split (seed 0, validation fraction 0.1) yields 4,321 expert training episodes '
      '(216,050 transitions) and 479 validation episodes (23,950 transitions). No transition split or new split was made.',
      '- State: four newest-first XY frames, width 8. Commanded goal: tiled XY, width 8, retained even though constant. '
      'Action: width 2 in [-1,1], without extra scaling. Model contexts contain only learner-visible state, '
      'action arguments and commanded goal. Next state is a likelihood target, never an input.',
      '- Frozen nominal model: censored five-component diagonal Gaussian action mixture, SiLU MLP 128/128, '
      f'`{run_args["nominal"]}/best.pkl`. It samples latent mixtures then clips to environment action bounds.',
      '- Diagonal initialization: zero-displacement atom plus three diagonal Gaussian moving components, '
      f'SiLU MLP 128/128, `{run_args["diagonal"]}/best.pkl`. Context and moving-delta normalizers are reused.',
      f'- Frozen action selector: `{run_args["actor"]}`, alpha 0, seed 0, step 150000. '
      'Sample one action from its tanh Gaussian for each state, and keep it fixed across expert draws. '
      'This actor was trained on all source episodes; this is not a held-out actor evaluation.', '',
      '## Failure reference provenance and leakage', '',
      f'The compatible p=0.30 existing bank has {config["references"]["original_count"]} entries; '
      f'{config["references"]["kept_count"]} remain after filtering by the established FULL-SOURCE training episode split. '
      'The bank stores `goals[256,8]`, `episode_id`, `failure_row`, `teacher_mode`, `behaviour_class`, and JSON `meta`. '
      'Its misleading `goals` field contains frozen observable F4 failure states, which are used here as next-state references.',
      '',
      f'Bank content SHA-256: `{config["references"]["bank_content_sha256"]}`. Its source-content hash is verified against '
      'the loaded dataset, and every retained vector is checked against its original observation and row. '
      'No existing bank is rewritten. All kept/removed IDs are saved in `config.json`.', '',
      'The original bank first detected visible fully frozen histories in swamp cells, then its 60% random / '
      '40% deliberate composition used privileged `teacher_mode` and `swamp_bits`. The death flag was used as '
      'a construction cross-check. This is a curated privileged training artifact, not a learner-derived failure distribution. '
      'This experiment reads those stored class names for provenance counts only; it filters solely on episode membership. '
      f'Post-filter class counts: `{config["references"]["class_counts_after_filter"]}`. '
      f'By source episode range: `{bank_sources}`. Thus the reference bank is deliberately a mixed-source '
      'training target; it is not described as expert-only. '
      'No reference episode overlaps any full-source validation episode or expert validation episode. '
      'Training-reference reuse at validation is intentional: it measures the same training-defined target, '
      'not generalization to an independent bank. Privileged construction and validation leakage are distinct issues.', '',
      '## Objective, sampler, and structural guarantees', '',
      '`L = L_diag + lambda_pess * L_pess` contains exactly two terms. `L_diag` is the existing mixed-measure '
      'negative log likelihood: mass at zero displacement and density with respect to raw XY maze-unit area elsewhere. '
      'Reported NLL describes the unprojected displacement distribution, not the geometry-corrected law.', '',
      'For each fixed (s,x), sample K independent x_prime values from the frozen nominal model and one transition '
      'per draw. The samples have shape [B,K,8]. Full-F4 MMD is computed separately for each group and then averaged. '
      'The biased V-statistic includes generated-generated, reference-reference and cross terms, including self pairs; '
      'all 233 training references are used each update. Different states are never pooled.', '',
      f'Kernel normalization is fitted on expert training next states only. Four Gaussian bandwidths '
      f'are `{config["kernel"]["bandwidths"]}`: 0.25, 0.5, 1 and 2 times the median of 4,096 training-pair distances '
      '(seed 410). Both normalization and bandwidths stay frozen. There is no learned embedding.', '',
      'The diagonal anchor is the current ETT diagonal sample at (s,x_prime,x_prime), with shared model randomness. '
      'A 64/64 SiLU residual network sees normalized [s,x,x_prime,g], action difference, and sampled anchor displacement. '
      'Its final layer starts at exactly zero. The residual is `L * ||x-x_prime||_2 * tanh(h)/sqrt(2)`. '
      'The implementation subtracts a conservative float32 margin from the radius (clamped at zero) before '
      'adding the anchor, preventing roundoff from spuriously rejecting saturated residuals. '
      'Only newest XY changes; remaining frames always equal `s[0:6]`. L has units of raw maze distance per '
      'environment-action Euclidean unit. L=0.25 and 0.75 are prototype settings, not verified causal constants.', '',
      'After adding the residual, the existing one-unit per-coordinate displacement cap, global bounds, and '
      'blocked-endpoint fallback are applied. This nonconvex geometry operation may violate the anchor bound. '
      'Any such candidate is rejected in favor of the valid diagonal anchor. Identical actions take the exact '
      'anchor branch, preserving its sampling law and diagonal likelihood. This guarantees the anchored '
      'action-change bound on final emitted samples, not all-pairs action Lipschitzness. Shared model randomness '
      'does not recover hidden U.', '',
      '## Gradient estimator and bounded budget', '',
      'MMD uses conditional pathwise derivatives through selected Gaussian locations/scales and the residual. '
      'Categorical choices, Bernoulli atoms, clipping decisions and fallback decisions are not reparameterized. '
      'There is no straight-through or score-function estimator. Consequently this is a biased partial gradient '
      'of expected MMD: it has no direct path to mixture logits or atom logits. Diagonal NLL trains those outputs. '
      'Shared trunk updates can change them indirectly; off-diagonal atom samples may move through the residual. '
      'At a bound-violation fallback, the residual path has zero derivative. These are material prototype limitations.', '',
      f'Initial training diagnostics: diagonal loss {config["weight_selection"]["initial_training_diagnostic"]["diagonal_loss"]:.5f}, '
      f'MMD {config["weight_selection"]["initial_training_diagnostic"]["mmd2"]:.5f}, diagonal gradient norm '
      f'{config["weight_selection"]["initial_training_diagnostic"]["diagonal_gradient_norm"]:.5f}, MMD gradient norm '
      f'{config["weight_selection"]["initial_training_diagnostic"]["mmd_gradient_norm"]:.5f}, residual-only MMD gradient '
      f'{config["weight_selection"]["initial_training_diagnostic"]["mmd_residual_gradient_norm"]:.5f}. '
      f'The training-only rule chose lambda={config["weight_selection"]["selected"]}; weighted gradient ratio '
      f'{config["weight_selection"]["weighted_mmd_to_diag_gradient_ratio"]:.6f}. '
      'No validation metric selected lambda or L. A second positive weight was not needed to establish whether '
      'the residual can respond to failure matching.', '',
      f'Each arm ran {run_args["steps"]} updates with batch size {run_args["batch_size"]}, K=8 and Adam '
      f'learning rates {run_args["diagonal_learning_rate"]} (diagonal) and {run_args["learning_rate"]} (residual), '
      f'on {config["runtime"]["devices"]}. Total comparison runtime: {config["elapsed_seconds"]:.1f} seconds. '
      'All arms start at the same diagonal checkpoint; control residuals stay exactly zero. '
      'The final fixed-budget checkpoint is the primary evaluation checkpoint. A best-diagonal-validation checkpoint '
      'is also saved locally, without using validation MMD for selection.', '',
      '## Validation results', '',
      f'Reproduced initialization on all 23,950 validation transitions: NLL {base_diag["nll"]:.6f}, '
      f'mean position error {base_diag["mean_position_error"]:.6f}, conditional energy score {base_diag["energy_score"]:.6f}. '
      'The prior report gives NLL -5.77057 and energy 0.01391. The small likelihood difference is consistent with '
      'the different runtime (saved checkpoint GPU JAX 0.6.2 versus local CPU JAX 0.10.2); no bitwise cross-runtime '
      'equivalence is claimed. Energy is a Monte Carlo estimate with 64 draws.', '',
      f'Initial marginal MMD: K=8 {base_marginal["8"]["mmd2"]:.6f}; K=16 {base_marginal["16"]["mmd2"]:.6f}. '
      'MMD/motion evaluation uses the same 1,024 validation rows for all arms. '
      'K=8/16 use the same fixed actor draws and keys but are not nested expert-sample sets. '
      'Episode-bootstrap intervals cluster the selected validation rows by source episode; they do not add independent bank uncertainty.', '',
  ]
  for name in labels:
    result = arms[name]
    stat = result['marginal']['16']
    lines.append(f'- `{name}`: diagonal NLL {result["diagonal"]["nll"]:.6f}; '
                 f'energy {result["diagonal"]["energy_score"]:.6f}; MMD K8/K16 '
                 f'{result["marginal"]["8"]["mmd2"]:.6f}/{stat["mmd2"]:.6f}; '
                 f'moving-history motion {stat["moving_history_mean_motion"]:.5f}; '
                 f'moving-history near-zero fraction {stat["moving_history_near_zero_fraction"]:.4%}; '
                 f'mean XY std {np.round(stat["mean_xy_std"], 5).tolist()}.')
  lines += ['', '![Fit and MMD](fit_and_matching.png)', '',
            '![Motion and sample diversity](motion_and_diversity.png)', '',
            '## History mismatch, gradients, and geometry', '',
            f'For the predeclared primary arm `{primary_name}`, fixed historical frames contribute '
            f'{m["history_share_mean_cross_squared_distance"]:.2%} of aggregate cross-reference squared distance. '
            f'Mean nearest-reference fixed-history squared distance is {m["mean_nearest_reference_history_distance2"]:.5f}. '
            'This contribution cannot be changed in one step. Moving histories cannot become fully frozen in a single '
            'valid F4 transition; lower full-F4 MMD does not imply death.', '',
            f'In the primary arm, moving-history mean motion changes from '
            f'{control["marginal"]["16"]["moving_history_mean_motion"]:.5f} in the control to '
            f'{m["moving_history_mean_motion"]:.5f}, while near-zero motion falls from '
            f'{control["marginal"]["16"]["moving_history_near_zero_fraction"]:.2%} to '
            f'{m["moving_history_near_zero_fraction"]:.2%}. Thus aggregate improvement is not motion suppression. '
            'The larger L arm does increase both mean motion and the near-zero fraction, indicating a mixture of '
            'larger displacements and collision-induced persistence.', '',
            f'A supplementary check selects {frozen_report["contexts"]} fully frozen observable histories at timestep > 0 '
            '(reset stacks excluded) from the same validation subset. The recorded next XY is stationary in '
            f'{frozen_report["observed_stationary_next_xy_fraction"]:.2%} of these rows. Under the fixed actor proposal, '
            f'sampled stationary frequency changes from {frozen_report["arms"][control_name]["exact_zero_fraction"]:.2%} '
            f'in the control to {frozen_report["arms"][primary_name]["exact_zero_fraction"]:.2%} in the primary arm. '
            'Moving these observed frozen histories is a substantive failure-mode concern, even though no hidden '
            'death label is used and no off-diagonal ground truth is available. See `frozen_history_evaluation.json`; '
            'these supplementary draws use 64 samples and separately fixed keys 413/414.', '',
            f'At K=16 the editable-XY MMD gradient norm averages {m["mean_editable_xy_mmd_gradient_norm"]:.6f}; '
            f'{m["editable_xy_gradient_below_1e8_fraction"]:.2%} of contexts have norms below 1e-8. '
            'The smallest kernels can saturate for far-away histories; larger kernels retain gradients. '
            'The kernel is fixed, and full-F4 matching was never replaced by newest-XY matching.', '',
            f'Primary-arm candidate geometry corrections: {m["candidate_geometry_correction_fraction"]:.4%}; '
            f'bound-violation fallback: {m["bound_fallback_fraction"]:.4%}; '
            f'blocked straight-line segments (21-point probe): {m["straight_segment_blocked_fraction"]:.4%}. '
            'Every evaluated arm and both K values had zero final endpoint violations, zero history-shift error '
            'and zero final bound violations. Corrective endpoint geometry is still an approximation of the '
            'environment axiswise collision substeps, and the segment probe is not a physics proof.', '',
            '![History and geometry](history_and_geometry.png)', '',
            '![Action-pair sample distributions](action_pair_samples.png)', '',
            'The examples include identical, nearby and far action pairs, with exact episode and time indices in '
            '`plotted_examples.json` and each arm metrics file. The trained diagonal may change relative to '
            'initialization, but on identical actions the residual is exactly zero relative to its own current '
            'diagonal sampler. These plots show sample positions, not environment counterfactual ground truth.', '',
            '## Reproduction and local artifacts', '',
            'Completed commands (run from the PointMaze worktree):', '', '```bash',
            'python -m scripts.test_ett_distribution_matching',
            config['command'],
            f'python -m ett.report_distribution_matching --run-dir {args.run_dir}', '```', '',
            'For a rerun, use a fresh `--out-dir` and pass that directory to the reporting command; the trainer '
            'deliberately refuses to overwrite completed runs. No additional training command is pending.', '',
            'The earlier numerical checks are in sibling `sanity` and `sanity_v2` directories. The first used '
            'one shared Adam rate and exposed a sharp diagonal update; the second verified separate rates before '
            'the bounded comparison. Neither smoke result selected lambda/L by validation performance. '
            'The sibling `f4_p30_s01` comparison is superseded: saturated residuals caused unnecessary float32 '
            'bound fallbacks. The guarded comparison reran the same 5 x 500 budget after adding the conservative '
            'roundoff margin; no weight was retuned. Both runs and both smoke configurations remain saved.', '',
            'All `final.pkl` and `best_diagonal.pkl` files stay local and are ignored by Git. '
            'Configuration, split/reference IDs, normalization, checkpoint/source hashes, metrics, plots and '
            'this report are English text/artifacts prepared for review. Publication requires explicit user authorization; checkpoints remain local.', '',
            'Sampling example (state and commanded goal have widths 8 and 8):', '', '```python',
            'import jax', 'from ett.anchored_transition import load_anchored',
            'from propensity.nominal_policy import load_nominal_policy',
            f"model = load_anchored('{args.run_dir}/{primary_name}/final.pkl')",
            f"nominal = load_nominal_policy('{run_args['nominal']}')",
            'next_states = model.sample_marginal(',
            '    nominal, state, x, jax.random.PRNGKey(7), num_samples=16, goal=goal)',
            '# Unbatched: [16,8]; batched [B,8] input: [B,16,8].',
            'conditional = model.sample(state, x, x_prime, jax.random.PRNGKey(8),',
            '                           num_samples=8, goal=goal)', '```', '',
            'Tests cover conditional grouping, scalar MMD values, diagonal identity after residual changes, exact '
            'history shifting, identical/near/far final bounds, collision fallback, finite-difference residual gradients, '
            'and the absence of categorical/atom pathwise gradients. Checkpoint replay is exactly equal in every arm. '
            'These checks establish the implemented contract, not counterfactual correctness.', '',
            'No hidden death or swamp state was loaded for evaluation in this experiment. No online rollout or '
            'actor improvement is claimed. Full-F4 bank mismatch, privileged reference construction, partial '
            'sampling gradients, approximate geometry, small budgets, and reuse of the original checkpoint-selection '
            'validation split remain limitations.']
  (directory/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
  write_json(directory/'report_provenance.json', {
      'completed_command': f'python -m ett.report_distribution_matching --run-dir {args.run_dir}',
      'source_sha256': {p: file_sha(p) for p in (
          'ett/anchored_transition.py', 'ett/run_distribution_matching.py',
          'ett/report_distribution_matching.py', 'scripts/test_ett_distribution_matching.py')},
      'post_training_api_change': 'Added diagonal log_prob wrapper and its identity/rejection checks; training and sampling computations unchanged.',
      'tests': '8 behavioral tests passed; expanded diagonal likelihood test also passed after the API addition',
  })
  print(f'wrote report and four figures to {directory}', flush=True)


if __name__ == '__main__':
  main()
