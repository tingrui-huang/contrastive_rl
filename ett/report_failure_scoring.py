"""Render saved scope-B feasibility results; no fitting or model updates."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve

from ett.run_failure_scoring import read
from ett.run_distribution_matching import write_json
from scripts.make_swamp_f4_failure_bank import file_sha

KINDS = ('xy', 'motion', 'f4')
COLORS = {'xy': '#997019', 'motion': '#267eab', 'f4': '#994ab0'}


def figures(out, metrics, training, examples, sensitivity):
  with np.load(out/'observed_test_predictions.npz',allow_pickle=False) as d:
    y = d['audit_dead']
    fig, axes = plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    for kind in KINDS:
      p = d['score_'+kind+'_s0']
      fpr,tpr,_=roc_curve(y,p)
      precision,recall,_=precision_recall_curve(y,p)
      axes[0].plot(fpr,tpr,label=kind.upper(),color=COLORS[kind])
      axes[1].step(recall,precision,where='post',label=kind.upper(),color=COLORS[kind])
      bins=metrics[kind+'_s0']['all']['calibration']['bins']
      axes[2].plot([b['predicted'] for b in bins],[b['observed'] for b in bins],'.-',label=kind.upper(),color=COLORS[kind])
    axes[0].set(xlabel='False-positive rate',ylabel='True-positive rate',title='Scorer holdout ROC')
    axes[1].set(xlabel='Recall',ylabel='Precision',title='Scorer holdout precision-recall')
    axes[1].axhline(y.mean(),color='gray',ls='--')
    axes[2].plot([0,1],[0,1],'--',color='gray')
    axes[2].set(xlabel='Mean sigmoid score',ylabel='Observed death fraction',title='Observed-data reliability (10 bins)')
    for ax in axes: ax.legend()
    fig.suptitle('Scope B: privileged-label supervised feasibility, seed 0')
    fig.savefig(out/'observed_roc_pr_reliability.png',dpi=160);plt.close(fig)
  groups=['dead_age_0_first_observation','dead_age_1','dead_age_2','dead_established_age_ge_3',
      'alive_zero_recorded_action_waiting','alive_nonzero_action_near_stationary_collision_compatible','alive_reset_seeded_equal_stack']
  labels=['Death age 0','Death age 1','Death age 2','Death age 3+', 'Alive zero-action','Alive collision proxy','Alive reset']
  fig,axes=plt.subplots(1,2,figsize=(14,5),constrained_layout=True)
  for k,kind in enumerate(KINDS):
    values=[metrics[kind+'_s0'][g]['positive_prediction_episode_interval'] for g in groups]
    y=np.array([v['mean'] for v in values]);ci=np.array([v['ci95_episode'] for v in values])
    axes[0].errorbar(np.arange(len(groups))+(k-1)*.18,y,yerr=np.maximum(0,np.stack([y-ci[:,0],ci[:,1]-y])),fmt='o',capsize=3,label=kind.upper(),color=COLORS[kind])
    for seed in (0,1):
      h=training[f'{kind}_s{seed}']['history']
      axes[1].plot([r['step'] for r in h],[r['selection_bce'] for r in h],label=f'{kind} seed {seed}',color=COLORS[kind],ls='-' if seed==0 else '--')
  axes[0].set_xticks(np.arange(len(groups)),labels,rotation=28,ha='right')
  axes[0].set(ylabel='Fraction with score >= 0.5',title='Seed 0, 95% episode-bootstrap intervals',ylim=(-.03,1.03))
  axes[1].set(xlabel='Update',ylabel='Unweighted BCE',title='Fixed training budget; selection split')
  axes[0].legend();axes[1].legend();fig.suptitle('Scope B: early death and stationary non-failure cases')
  fig.savefig(out/'difficult_strata_and_training.png',dpi=160);plt.close(fig)
  candidate=read(out/'controlled_candidate_scores.json')['scores']
  names=['persistence','diagonal_only','mmd_trained','symmetric_jitter_radius_0.05','history_nearest_collapse_projected']
  display=['Persistence','Diagonal-only','MMD-trained','0.05 jitter','Projected collapse']
  fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
  for ax,group in zip(axes,['confirmed_established_dead_context','alive_context']):
    for kind in KINDS:
      ax.plot(display,[candidate[n][kind+'_s0'][group]['score']['mean'] for n in names],'.-',label=kind.upper(),color=COLORS[kind])
    ax.set(ylabel='Mean candidate score',title=group.replace('_',' '),ylim=(0,1))
    ax.tick_params(axis='x',rotation=25);ax.legend()
  fig.suptitle('Scope B: synthetic outputs have no ground-truth failure labels')
  fig.savefig(out/'controlled_candidate_scores.png',dpi=160);plt.close(fig)
  with np.load(out/'gradient_plot_arrays.npz',allow_pickle=False) as data:
    fig,axes=plt.subplots(2,3,figsize=(13,8),constrained_layout=True)
    titles=['Established audited death (age >=3)', 'Audited death onset (age 0)',
            'Audited death age 1', 'Alive zero-action waiting proxy',
            'Alive collision-compatible low motion', 'Alive reset stack']
    for i,ax in enumerate(axes.ravel()):
      n=i*8;ex=examples[n];xy=data[f'xy_{n}'];p=data[f'score_{n}'];valid=data[f'valid_{n}']
      contours=ax.tricontourf(xy[:,0],xy[:,1],p,levels=np.linspace(0,1,21),vmin=0,vmax=1,cmap='viridis')
      ax.scatter(xy[~valid,0],xy[~valid,1],s=6,c='red',marker='x',alpha=.4)
      base=np.array(ex['observed_state'])[:2];g=np.array(ex['models']['f4_s0']['newest_xy_gradient'])
      arrow=.07*g/max(np.linalg.norm(g),1e-10)
      ax.plot(base[0],base[1],'wo',mec='black');ax.arrow(*base,*arrow,width=.003,color='white',length_includes_head=True)
      ax.set(title=f'{titles[i]}\nep {ex["episode_id"]}, row {ex["observation_row"]}',xlabel='Newest X',ylabel='Newest Y',aspect='equal')
      ax.title.set_fontsize(9)
    fig.suptitle('Scope B F4 seed 0: score contours with old frames fixed\nWhite arrow: rescaled ascent; red crosses: failed geometry checks; synthetic labels unknown')
    fig.colorbar(contours,ax=axes.ravel().tolist(),label='Sigmoid score',shrink=.8)
    fig.savefig(out/'local_score_landscapes.png',dpi=160);plt.close(fig)
  fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
  for kind in KINDS:
    selected=[e for e in examples if e['group']=='dead_established_age_ge_3']
    axes[0].plot(np.arange(len(selected)),[e['models'][kind+'_s0']['grid_score_gain'] for e in selected],'.-',label=kind.upper(),color=COLORS[kind])
    axes[1].plot(np.arange(len(sensitivity['states'])),sensitivity['scores'][kind+'_s0'],'.-',label=kind.upper(),color=COLORS[kind])
  axes[0].set(xlabel='Distinct confirmed-death episode example',ylabel='Best local-grid score minus observed score',title='Can movement improve an absorbing-state score?')
  axes[1].set(xlabel='Constant-stack coordinate sweep index',ylabel='Sigmoid score',title='Location / invalid-state sensitivity; labels unknown')
  for ax in axes:ax.legend()
  fig.suptitle('Scope B: optimization-target probes, not causal validation')
  fig.savefig(out/'optimization_and_sensitivity.png',dpi=160);plt.close(fig)


def write_report(out, c, m, training, examples, sensitivity):
  lines=['# Scope-B PointMaze failure-scoring feasibility', '',
      '**Decision B + D: useful for established observed failures, insufficient as a justified one-step failure objective; supervision is privileged.**', '',
      'This is a supervised feasibility probe, not an offline-compatible failure reward, diagonal/off-diagonal discriminator, counterfactual validator, or worst-case value function. '
      'No ETT, actor or critic was updated. The fixed scorer comparisons and synthetic audits below do not validate optimizing a transition model against these scores.', '',
      '## Label and environment audit', '',
      f'Environment `{c["dataset"]["env_name"]}`, per-cell active probability 0.30, 6,600 episodes of 50 transitions. Dataset SHA-256 '
      '`83b4e81d9fca2d66b648c9c34ccdb68196e1acf89e2ca6829e4cb198d27c322c`. '
      'The authoritative guarded distribution run and objective diagnostic at branch commit `71a7219` were reused; no newer remote commit existed at task start.', '',
      'The label is **dead when the scored observation is seen**, reconstructed from the existing `entered_active_swamp` per-episode audit flag, '
      'the pre-action `swamp_bits`, and the next landed cell. The environment moves, checks the current mask, sets its absorbing death flag, then redraws the mask; '
      'the F4 wrapper pushes newest XY after physics. Fatal transition row d-1 is alive; observation d has death age 0. '
      'The previous observability reconstruction and existing ETT audit agree exactly. Death labels are never inferred from stillness. '
      'The early ages 0/1/2 retain old motion; an equal four-frame history forms at age 3 in observed death episodes.', '',
      'Scorer rows follow the established valid-transition view: observations t=0..49, excluding the terminal observation/dummy action row. '
      'Death at the last transition can therefore be recorded at episode level without a positive scorer row; this is explicit horizon censoring.', '',
      'Death is absorbing with ignored actions and zero reward until the fixed horizon; `done` is not a death indicator. '
      'Reset repeats an alive position. Zero recorded action is a waiting/holding proxy, not a death label; action noise can still move it. '
      'The collision stratum is alive, action norm >0.05 and observed next-motion norm <=0.05. No collision flag or physics-noise trace was recorded, '
      'so these are collision-compatible examples, not confirmed collisions. Exact collision-specific performance remains unmeasured. '
      'Actions and future motion are used only for audit strata, not model inputs.', '',
      'Uniform coverage [0,1200), teacher/expert-positive source [1200,6000), and appended blind demonstrator [6000,6600) '
      'are source memberships, not death/non-death or quality labels. Contrastive positive/negative labels express the earlier contrastive sampling task, '
      'not simulator survival. Non-success and absence from a failure bank supply no binary death target. '
      'The bank stores 233 retained training references from three sources; its original curated mixture used privileged teacher modes and swamp masks, '
      'with death cross-checks. Its stored F4 states alone do not provide reliable alive labels or onset timing. '
      'Only scope B has reliable binary targets under the present files: the collector explicitly designates these audit fields unavailable to learner training.', '',
      '## Population, independent scorer partitions and reuse', '',
      'All three existing source populations are included for this supervised diagnostic. This expands the earlier expert-only observability probe, '
      'adds a motion-only ablation, separates selection from scorer test, and tests the frozen scores on generated/controlled observations. '
      'The established seed-0 full-source 10% holdout becomes scorer test; its complement is split 90/10 using seed 9201 into scorer training and selection. '
      'Every complete episode stays in one partition; byte-identical complete observed trajectories crossing partitions are rejected.', '']
  for name,v in c['splits'].items():
    a=v['temporal_audit']
    lines.append(f'- {name}: {v["episodes"]:,} episodes / {v["rows"]:,} transitions; current-death prevalence {a["dead_prevalence"]:.2%}; '
        f'{a["dead_episode_count"]} observed death episodes; {len(v["bank_episode_overlap"])} retained-bank episode overlaps; '
        f'{len(v["prior_objective_development_episode_overlap"])} prior objective-development episode overlaps.')
  lines += ['', 'Test is independent of this scorer fitting/selection only. The actor already used every source episode; the expert holdout was '
      'previous observability checkpoint-selection data. The old 1,024 diagnostic contexts from 427 test episodes are development data and are explicitly '
      'separated from the other test episodes in metrics. A new partition does not erase upstream reuse. All scorer-test episodes are absent from the '
      'retained training bank; exact vector membership is also tested separately. Bank references are not supervised targets or model inputs.', '',
      '## Inputs and fixed training budget', '',
      'Three 64/64 SiLU MLPs use (1) current XY; (2) six successive newest-first displacements `[p0-p1,p1-p2,p2-p3]`; '
      '(3) all eight F4 coordinates. Each retains the same commanded-goal block, which is constant and normalizes to zero. '
      'No action, future observation, reward, hidden field, episode ID, source code or death age enters any scorer. '
      'The motion ablation deliberately removes location; this is not an assumption that actual failure hazards are translation invariant.', '',
      'All models use train-only normalization, unweighted BCE, uniform transition minibatches of 1,024, Adam 3e-4, '
      '2,000 updates and seeds 0/1. This leaves the natural observed class prior unchanged. The lowest selection BCE at 250-update intervals '
      'selects each checkpoint. Test scores and synthetic audits never select a checkpoint, threshold or hyperparameter; threshold 0.5 is fixed. '
      'Raw sigmoid outputs are uncalibrated scores; observed-prevalence reliability/Brier diagnostics are reported, without post-hoc calibration '
      'or a claim of calibrated deployment probabilities.', '',
      '## Observed scorer-test results', '']
  for name in ('xy_s0','xy_s1','motion_s0','motion_s1','f4_s0','f4_s1'):
    v=m[name]['all']
    lines.append(f'- {name}: ROC-AUC {v["roc_auc"]:.5f}; average precision {v["precision_recall_auc_average_precision"]:.5f}; '
        f'Brier {v["brier"]:.5f}; BCE {v["binary_cross_entropy"]:.5f}; precision/recall '
        f'{v["dead_precision"]:.2%}/{v["dead_recall"]:.2%}; ECE {v["calibration"]["ece_10_equal_width"]:.5f}.')
  paired=read(out/'paired_episode_intervals.json')
  lines += ['', 'The full-F4 gain is meaningful overall, but not uniform across strata. For seed 0, the paired episode-bootstrap '
      f'F4-minus-XY ROC-AUC interval is {paired["f4_minus_xy_s0"]["all"]["roc_auc"]["ci95"]}; '
      f'F4-minus-motion is {paired["f4_minus_motion_s0"]["all"]["roc_auc"]["ci95"]}. '
      'The following distinctions prevent the aggregate gain from being interpreted as reliable onset recognition:', '',
      f'- XY is mainly a location detector: within the swamp corridor its ROC-AUC is '
      f'{m["xy_s0"]["region_swamp_corridor"]["roc_auc"]:.4f}, despite the high overall AUC. '
      f'It marks {1-m["xy_s0"]["alive_zero_recorded_action_waiting"]["specificity"]:.2%} of alive zero-action rows positive.',
      '- Motion-only marks all 660 alive reset stacks positive, and the single alive equal stack after reset. '
      'It cannot distinguish these from established dead stacks using motion alone. F4 marks these rows negative by also using location.',
      f'- Inside the swamp corridor, motion-only AUC/AP '
      f'{m["motion_s0"]["region_swamp_corridor"]["roc_auc"]:.4f}/'
      f'{m["motion_s0"]["region_swamp_corridor"]["precision_recall_auc_average_precision"]:.4f} '
      f'exceed F4 {m["f4_s0"]["region_swamp_corridor"]["roc_auc"]:.4f}/'
      f'{m["f4_s0"]["region_swamp_corridor"]["precision_recall_auc_average_precision"]:.4f}; F4 is not uniformly better.',
      f'- For onset versus alive, F4 AUC/AP are {m["f4_s0"]["onset_vs_alive"]["roc_auc"]:.4f}/'
      f'{m["f4_s0"]["onset_vs_alive"]["precision_recall_auc_average_precision"]:.4f}. '
      'Its paired onset AUC is lower than XY, whose apparent early recall comes with many location-related false positives. '
      'Later immobility accounts for most of the strong overall death discrimination.', '',
      'Late-window censoring is preserved: test contains 140 age-0, 139 age-1 and 137 age-2 observations. '
      'Episodes ending before a later age do not acquire invented rows or labels. The single alive equal stack after reset is a one-episode '
      'stratum and cannot establish general stationary-alive performance. The safe-route stratum has 1,099 rows from 88 episodes and no '
      'death positives; specificity there is not a death-recognition test.', '',
      'Source strata have different prevalences. Seed-0 F4 AP is '
      f'{m["f4_s0"]["source_uniform_coverage"]["precision_recall_auc_average_precision"]:.4f} for uniform coverage, '
      f'{m["f4_s0"]["source_expert_positive_source"]["precision_recall_auc_average_precision"]:.4f} for teacher source, and '
      f'{m["f4_s0"]["source_blind_demonstrator"]["precision_recall_auc_average_precision"]:.4f} for blind demonstrator. '
      'These source strata were represented during scorer fitting; this is unseen-episode generalization, not unseen-policy or unseen-region transfer. '
      'All 5,335 dead test rows are from episodes absent from the retained bank; their metrics are separately stored. '
      'Outside prior objective-development episodes there are 233 test episodes, still seen by the original actor. '
      'The older expert-only probe used 4,000 updates and a different population; its numbers are not a matched-budget control for this experiment.']
  lines += ['', '![ROC, PR and reliability](observed_roc_pr_reliability.png)', '',
      'Seed-0 F4 performance by difficult stratum (positive-prediction rate means recall on dead-only rows and false-positive rate on alive-only rows):', '']
  for group in ('dead_age_0_first_observation','dead_age_1','dead_age_2','dead_established_age_ge_3',
      'alive_zero_recorded_action_waiting','alive_nonzero_action_near_stationary_collision_compatible',
      'alive_reset_seeded_equal_stack','alive_equal_stack_after_reset','alive_moving_history'):
    v=m['f4_s0'][group]
    if v['count']:
      q=v['positive_prediction_episode_interval']
      lines.append(f'- {group}: {v["count"]:,} rows / {v["episodes"]} episodes; score >=0.5 rate {q["mean"]:.2%} '
          f'(95% episode interval [{q["ci95_episode"][0]:.2%}, {q["ci95_episode"][1]:.2%}]); mean score {v["mean_probability"]:.4f}.')
    else: lines.append(f'- {group}: no rows; no performance claim.')
  lines += ['', 'Single-class strata have undefined ROC-AUC/AP, explicitly stored as null. `onset_vs_alive` and `early_vs_alive` '
      'supply two-class comparisons; spatial, source, bank-membership, and prior-development reuse strata are all saved, including empty strata. '
      'Mean scores and threshold rates have 500-replicate episode-bootstrap intervals. F4-minus-XY and F4-minus-motion ROC-AUC/AP/Brier '
      'comparisons use 200 paired complete-episode bootstrap replicates, separately for all rows, swamp, onset-versus-alive and early-versus-alive. '
      'These intervals condition on the fitted model and source distribution, and do not create upstream-independent evidence.', '',
      '![Difficult strata and fitting](difficult_strata_and_training.png)', '',
      '## Frozen-score optimization-target audit', '',
      'Both scorer seeds evaluate the saved seed-0 diagonal-only and MMD-trained K=16 outputs at the exact 1,024 prior contexts. '
      'Their original actor and nominal draws are preserved by reading the saved samples. Synthetic outputs have no ground-truth failure labels. '
      'Context death labels are aligned from audit records and apply to the source context only. Persistence, cardinal 0.01/0.05 perturbations, '
      'symmetric 0.05 jitter and raw/projected history-nearest collapse exactly follow the prior diagnostic constructions. '
      'All old frames are preserved. Endpoint, per-coordinate cap, sampled segment and original sampled anchor-bound checks are reported separately; '
      'a constructed candidate may violate the anchor bound. These checks do not prove physical reachability.', '',
      'Audit labels independently confirm that the 91 frozen non-reset contexts are established dead states. '
      'Seed-0 F4 mean scores for persistence / diagonal-only / MMD-trained / projected collapse are '
      '0.96052 / 0.94526 / 0.95543 / 0.96559. Thus the scorer ranks the MMD-trained outputs above the '
      'diagonal-only outputs on average even though their sampled motion is much larger (0.3493 versus 0.0516 maze units). '
      'It also rewards moving to the projected reference location over persistence on average. '
      'Both model-output sets pass the saved anchor check here; 27.54% of projected-collapse draw/radius pairs fail it. '
      'The collapse comparison is therefore structural evidence, not a feasible-improvement claim for every draw.', '',
      '![Controlled synthetic candidate scores](controlled_candidate_scores.png)', '',
      'Gradient examples use the first eight distinct scorer-test episodes in each of six named strata, independent of predicted scores. '
      'They perturb the scored observed state itself, retaining its old six coordinates. Geometry is checked relative to the preceding source observation; '
      'a matching action-pair anchor is unavailable for these examples and no anchor guarantee is asserted. '
      'We report newest-XY derivatives, a projected 0.05 ascent step, and the highest-scoring endpoint/cap/segment-valid candidate in a fixed '
      '21x21 grid extending 0.25 maze units around the observed XY. This is an input sensitivity audit, not a transition-model optimization run.', '']
  for kind in KINDS:
    selected=[e['models'][kind+'_s0'] for e in examples if e['group']=='dead_established_age_ge_3']
    lines.append(f'- {kind} seed 0, {len(selected)} confirmed established-death examples: mean local-grid score gain '
        f'{np.mean([v["grid_score_gain"] for v in selected]):.5f}; '
        f'{sum(v["grid_score_gain"]>1e-4 for v in selected)}/{len(selected)} have gain >1e-4; '
        f'mean maximizing XY displacement {np.mean([v["grid_best_motion_from_observed"] for v in selected]):.4f} maze units.')
  lines += ['', '![Local score landscapes](local_score_landscapes.png)', '',
      'The eight displayed established-death examples come from the earliest eligible uniform-coverage episodes, '
      'so their counts are qualitative examples, not population estimates. In both scorer seeds, every one gains score '
      'by moving within the valid local grid; the mean gain is about 0.029/0.027 and the selected movement about 0.256/0.255 maze units. '
      'Even the 0.05 projected gradient step raises mean F4 score by about 0.0065/0.0061. '
      'These are actual audited absorbing observations, so the incentive to move their newest coordinate is an optimization-target failure. '
      'An available action-bound guarantee was not established for these examples.', '',
      'Whole-history translations preserve displacement features but may change true hazard exposure. Their score changes are only location sensitivity, '
      'not accuracy under label-preserving augmentation. A separate constant-stack coordinate sweep includes wall and out-of-maze points, with '
      'unknown failure labels. Approximate F4 support distance uses 12,000 training reference rows and a 99th-percentile threshold from 4,000 '
      'rows in disjoint internal training episodes; it is neither a physical validity proof nor a density estimate. '
      'Confident scores on unsupported/invalid synthetic observations are reported explicitly. No test or synthetic outcome tunes these settings.', '',
      'The fixed coordinate sweep does **not** show >0.9 scores on any of its 15 invalid endpoints for either F4 seed. '
      'For seed 0 the largest invalid-point score is 0.573. This negative result bounds this particular probe, not all adversarial inputs. '
      'Motion-only assigns every constant stack approximately 0.842 regardless of whether it is an alive reset, a swamp failure or invalid geometry. '
      'On observed scorer-test rows, 210/33,000 exceed the training-derived support threshold (0.8832); '
      '3 of those 210 receive F4 scores above 0.9 in each seed. They are observed examples with audit labels, '
      'whereas the saved synthetic confidence/support results have no automatic truth labels.', '',
      '![Optimization and coordinate sensitivity](optimization_and_sensitivity.png)', '',
      '**Calibration on observed data does not establish reliability on optimized synthetic states.** '
      'Strong recognition of prolonged immobility at hazard locations can coexist with weak failure-onset evidence, false alarms on normal '
      'waiting/collision-compatible rows, and exploitable score gradients. A synthetic observation with a higher score is not validated as more failed.', '',
      '## Decision and one recommended next experiment (not run)', '',
      'The intended later loss `-E_{x_prime~b, y~T}[C_fail(y)]` was not implemented. '
      'The supervised results do not supply reliable offline-compatible labels or justify a one-step pessimistic transition objective. '
      'The most defensible use under this dataset is a separately marked audit score for established failures; early post-contact states need distinct treatment. '
      'This experiment does not establish fundamental F4 non-identifiability or solve worst-case Q.', '',
      'Recommended next: preregister a **multi-step observed-outcome feasibility comparison** with horizon 1 versus 4, using the same '
      'episode partitions and privileged audit protocol, without ETT or policy optimization. Fix the current scorer and assess the sequence of '
      'scores across actual onset trajectories and matched alive waiting/collision-compatible trajectories. Evaluate detection delay, early recall '
      'at a selection-chosen false-positive constraint, and source/spatial breakdowns; reserve a separately collected future simulator test set '
      'before claiming independent generalization. Existing audit supervision remains explicitly diagnostic. This would test whether allowing '
      'observable consequences to accumulate helps outcome assessment, not whether a generated transition is causally correct. No such follow-up was launched.', '',
      '## Provenance, reproduction and completion', '',
      'All exact partition/source episode IDs, data/bank/model hashes, input definitions, normalizers, loss histories and budgets are saved in '
      '`config.json` and `training.json`; prior files are hash-checked unchanged. Local checkpoints are named `supervised_probe_*` and declare '
      '`B_supervised_privileged_label_feasibility_only`, `offline_compatible=false`. They are ignored by Git. '
      'Publication of the code, configuration, metrics and evaluation artifacts was separately authorized by the user. '
      'Checkpoints remain local and are excluded from publication.', '',
      f'Completed: five independent contract tests, three updates for all three models and both seeds, '
      f'the fixed 2,000-update comparison, observed stratified/paired/calibration evaluation, frozen synthetic scoring, local gradients and support/geometry audits. '
      f'Full execution took {c["elapsed_seconds"]:.1f} seconds on {c["runtime"]["devices"]}. '
      'Unrun/unavailable: true collision labels, an upstream-untouched dataset, off-diagonal counterfactual outcome truth, deployment calibration, '
      'ETT loss optimization, A/B/C comparisons and any actor-critic update.', '',
      '```bash', 'python -m scripts.test_failure_scoring',
      'python -m ett.run_failure_scoring --out-dir artifacts/failure_scoring/scope_b_three_update --steps 3 --seeds 0,1 --eval-every 1 --numerical-only',
      c['command'],f'python -m ett.report_failure_scoring --run-dir {out.as_posix()}', '```', '',
      'Use a fresh output directory when reproducing. The dataset and existing guarded-run dependencies are resolved from the preserved '
      'source configuration; a missing dependency fails explicitly rather than being regenerated.', '',
      'The three-update check preceded addition of the synthetic audit and extra reporting strata; its configuration preserves the earlier '
      'driver-source hash. The full run records the final training/audit driver hashes. No probe architecture or training setting was tuned from the smoke result.', '',
      '```python', 'import jax.numpy as jnp', 'from ett.failure_scoring import load_failure_scorer',
      f'scorer = load_failure_scorer("{out.as_posix()}/supervised_probe_f4_s0.pkl")',
      '# Visible newest-first F4 only; the saved constant commanded goal is used.',
      'score = scorer.score(jnp.asarray([[4.0, 3.5] * 4]))  # shape [1]',
      '# Scope B: privileged-supervision diagnostic, not a deployment probability.', '```']
  (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
  p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',required=True);args=p.parse_args()
  out=Path(args.run_dir);c=read(out/'config.json')
  if c['status']!='complete':raise ValueError('diagnostic is not complete')
  m=read(out/'observed_test_metrics.json')['models'];training=read(out/'training.json')
  examples=read(out/'observed_gradient_examples.json')['examples'];sensitivity=read(out/'synthetic_sensitivity.json')
  figures(out,m,training,examples,sensitivity)
  write_report(out,c,m,training,examples,sensitivity)
  write_json(out/'report_provenance.json',{'scope':c['scope'],'source_sha256':file_sha(__file__),
      'command':f'python -m ett.report_failure_scoring --run-dir {out.as_posix()}', 'no_model_updates_or_draws':True})
  print('wrote scope-B report and five figures',flush=True)


if __name__=='__main__':main()
