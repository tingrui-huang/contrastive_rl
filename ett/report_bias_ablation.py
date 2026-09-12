"""Compare saved four-weight and six-parameter results without new sampling."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from ett.bias_ablation import load_arrays
from ett.action_response import GROUPS
from ett.run_rollout_return import read
from ett.run_distribution_matching import write_json
from scripts.make_swamp_f4_failure_bank import file_sha


NAMES=('control','six_s0','four_s0','six_s1','four_s1')
LABELS=('Control','Six, seed 0','Four, seed 0','Six, seed 1','Four, seed 1')
COLORS=('#657580','#dcaa85','#b84918','#94b7df','#2459a0')


def describe(reset,probe):
  reward=reset['reward'];hit=reward.any(-1);first=reward.argmax(-1)+1
  direction=probe['direction'][:,:9];valid=probe['radius'][:,:9]>1e-6
  mean=(direction*valid[...,None]).sum(1)/np.maximum(valid.sum(1)[...,None],1)
  specific=np.sqrt((((direction-mean[:,None])**2)*valid[...,None]).sum()/max(valid.sum(),1))
  return {'first_reward_step_conditional_on_any_reward':float(first[hit].mean()),
      'direction_action_specific_rms':float(specific),
      'direction_variation_across_anchor_samples_rms':float(np.sqrt(np.mean(np.sum((direction-direction.mean(2,keepdims=True))**2,-1)))),
      'raw_residual_near_zero_fraction':float((np.linalg.norm(probe['proposal_xy'][:,:9]-probe['anchor_xy'][:,:9],axis=-1)<.001).mean())}


def figures(out,evaluation,metrics,resets,probes,extra):
  fig,axes=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
  axes[0].bar(range(5),[evaluation['reset'][n]['model_return_mean'] for n in NAMES],color=COLORS)
  axes[0].set_xticks(range(5),LABELS,rotation=35,ha='right');axes[0].set(ylabel='Predicted discounted hard return',title='512 matched evaluation paths per arm')
  for name,label,color in zip(NAMES,LABELS,COLORS):
    axes[1].plot(range(1,51),resets[name]['reward'].mean((0,1)),label=label,color=color,ls='--' if name.startswith('six') else '-')
  axes[1].set(xlabel='Model step',ylabel='Fraction receiving hard reward',title='Unchanged radius-2 reward');axes[1].legend(fontsize=8)
  for seed in (0,1):
    delta=evaluation['paired_four_minus_six'][f'four_s{seed}'];lo,hi=delta['ci95_paired_model_paths'];m=delta['mean']
    axes[2].errorbar(m,seed,xerr=[[m-lo],[hi-m]],fmt='o',color=COLORS[2+2*seed],capsize=4)
  axes[2].axvline(0,color='gray',ls='--');axes[2].set_yticks([0,1],['Seed 0','Seed 1'])
  axes[2].set(xlabel='Four-weight minus six-parameter return',title='Paired model-path 95% bootstrap intervals')
  fig.savefig(out/'returns_and_timing.png',dpi=160);plt.close(fig)
  names=NAMES[1:];labels=LABELS[1:];colors=COLORS[1:];x=np.arange(4)
  fig,axes=plt.subplots(1,4,figsize=(15,4),constrained_layout=True)
  axes[0].bar(x,[metrics[n]['all']['common_direction_energy_fraction']*100 for n in names],color=colors)
  axes[0].set(ylabel='Common direction energy (%)',ylim=(0,100),title='Direction remains mostly common')
  for offset,key,label in [(-.17,'mean_raw_residual_magnitude','Raw proposal'),(.17,'mean_emitted_anchor_displacement','Emitted')]:
    axes[1].bar(x+offset,[metrics[n]['all'][key] for n in names],width=.32,label=label,alpha=.8)
  axes[1].set(ylabel='Mean maze-unit displacement',title='Residual magnitude');axes[1].legend(fontsize=8)
  for offset,key,label in [(-.17,'half_opposing_x_response_norm','Opposing X'),(.17,'half_opposing_y_response_norm','Opposing Y')]:
    axes[2].bar(x+offset,[metrics[n]['all'][key] for n in names],width=.32,label=label,alpha=.8)
  axes[2].set(ylabel='Mean paired response norm',title='Action response also shrinks');axes[2].legend(fontsize=8)
  axes[3].bar(x,[metrics[n]['all']['candidate_corrected_fraction']*100 for n in names],color=colors)
  axes[3].set(ylabel='Corrected samples (%)',title='One-step geometry corrections')
  for ax in axes:ax.set_xticks(x,labels,rotation=40,ha='right')
  fig.savefig(out/'action_response_comparison.png',dpi=160);plt.close(fig)
  fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
  for row,group in enumerate(('observed_frozen_nonreset','observed_moving')):
    for col,key in enumerate(('mean_first_step_motion','model_return_mean')):
      ax=axes[row,col];ax.bar(range(5),[evaluation['continuation'][n][group][key] for n in NAMES],color=COLORS)
      ax.set_xticks(range(5),LABELS,rotation=35,ha='right');ax.set_title(f'{group}: {"first-step motion" if col==0 else "remaining model return"}',fontsize=10)
  fig.suptitle('Same 32 stationary and 32 moving observed histories; original remaining horizon only')
  fig.savefig(out/'continuation_comparison.png',dpi=160);plt.close(fig)
  groups=np.repeat(read(evaluation['action_context_file'])['groups'],2);action_colors=plt.get_cmap('tab10')(np.arange(10))
  fig,axes=plt.subplots(4,4,figsize=(13,11),constrained_layout=True);examples=[]
  for row,group in enumerate(GROUPS):
    index=int(np.flatnonzero(groups==group)[0]);examples.append({'group':group,'context_times_nominal_draw_index':index})
    for col,(name,label) in enumerate(zip(names,labels)):
      ax=axes[row,col];r=probes[name]
      emitted=(r['next_state'][index,:,:,:2]-r['anchor_xy'][index]).mean(1)
      raw=(r['proposal_xy'][index]-r['anchor_xy'][index]).mean(1)
      for a in range(10):
        ax.plot([0,raw[a,0]],[0,raw[a,1]],':',color=action_colors[a],alpha=.5)
        ax.plot(*raw[a],'x',color=action_colors[a],ms=4)
        ax.annotate('',xy=emitted[a],xytext=(0,0),arrowprops={'arrowstyle':'->','color':action_colors[a],'lw':1.})
      ax.plot(0,0,'k*',ms=5);ax.axhline(0,c='.8',lw=.5);ax.axvline(0,c='.8',lw=.5)
      ax.set(xlim=(-.32,.15),ylim=(-.23,.3),aspect='equal',title=f'{label}\n{group}')
      ax.title.set_fontsize(8)
      if col==0:ax.set_ylabel('Y change from anchor')
      if row==3:ax.set_xlabel('X change from anchor')
  action_labels=('Zero','Right .5','Left .5','Up .5','Down .5','Right 1','Left 1','Up 1','Down 1','Identity')
  fig.legend(handles=[plt.Line2D([0],[0],color=action_colors[i],label=label) for i,label in enumerate(action_labels)],loc='outside lower center',ncol=5,fontsize=8)
  fig.suptitle('First selected context per group, first nominal draw; same anchors and action colors\nCrosses/dotted lines: raw proposal; arrows: emitted displacement; star: anchor',fontsize=11)
  fig.savefig(out/'raw_and_emitted_examples.png',dpi=150);plt.close(fig)
  write_json(out/'example_selection.json',{'rule':'first selected context per group and first nominal action; no outcome selection','examples':examples})


def main():
  p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',required=True);args=p.parse_args();out=Path(args.run_dir)
  config=read(out/'config.json');evaluation=read(out/'evaluation.json');metrics=read(out/'action_response_metrics.json')
  if config['status']!='complete' or config['smoke']:raise ValueError('report requires completed full fixed-budget run')
  resets={n:load_arrays(evaluation['artifact_sources'][n]['reset']) for n in NAMES}
  probes={n:load_arrays(evaluation['artifact_sources'][n]['probe']) for n in NAMES}
  extra={n:describe(resets[n],probes[n]) for n in NAMES};write_json(out/'additional_diagnostics.json',extra)
  figures(out,evaluation,metrics,resets,probes,extra)
  lines=['# Bounded zero-bias ETT return-search ablation','',
    '**Conclusion: removing the two output biases is insufficient.** Both four-weight searches still reduce model return and retain overwhelmingly '
    'common residual directions. Residual magnitude decreases, but opposing-action response magnitude also decreases. This is partial suppression '
    'of the previous drift mechanism, without establishing successful learning of execution-action consequences. Stop further parameter-only tuning of this restricted '
    'search and do not begin actor-critic alternation.','',
    '## Preserved experiment and exact change','',
    'Reviewed `03f7686a2277c684d3801a6dcb7f24bd81ea622c` and `b2e65ad41dc6ed60525dee646d85cace18d22a00` on '
    '`feature/pointmaze-causal-transition`. The authoritative action diagnostic is '
    '`artifacts/ett_action_response/f4_return_s01_horizon50/REPORT.md`; its over-horizon predecessor is not used. '
    'The rejected raw-critic investigation was not repeated.','',
    'Initialize from the exact original `ett_distribution_matching/f4_p30_s01_guarded/s0_L0p25_lambda0/final.pkl`, whose output head is zero. '
    'The experiment does not remove biases from an already searched checkpoint. Map four coordinates to output weights at hidden rows 0 and 32; '
    'both output biases stay exactly zero, including at perturbation points. Diagonal parameters, residual torso and unselected weights remain frozen. '
    'The network architecture and anchor bound L=0.25 are unchanged. L is an engineering restriction, not an established causal constant.','',
    'Nominal and actor are frozen and keep their state-plus-goal conditioning. The task remains fixed-goal windy PointMaze F4 at p=0.30, '
    'newest-first state width 8, commanded goal `(8.5,3.5)` tiled four times and action width 2 bounded by [-1,1]. '
    'Each step independently draws nominal x_prime and execution action x before sampling the transition. Only x is stored as the execution action.','',
    '`L_ETT = L_diag + 1.0 * E[sum_{t=0}^{49} 0.95^t r(s_{t+1})]` is minimized, with exactly two terms. '
    'The reward is the unchanged strict radius-2 indicator evaluated on generated float32 visible XY; no goal termination, shaping, bank distance or '
    'critic substitution is introduced. The fixed-task hidden death gate is outside the reward region; finite-precision boundary caveats from the '
    'previous report still apply. No hidden fields are policy or reward inputs.','',
    f'**L_diag is constant, so this is restricted-search ablation, not joint two-loss training.** Train NLL '
    f'{config["diagonal_train_nll_constant"]:.8f}; held-out NLL {config["diagonal_validation_nll_constant"]:.8f} for all checkpoints. '
    'This is the unchanged mixed zero-atom / raw-XY-area diagonal likelihood in nats per transition, before geometry projection. '
    'The original complete-episode split and normalizers are reused.','',
    '## Fixed budget, common randomness and reuse','',
    'Seeds 0 and 1 each run 12 iterations, four iid N(0,I4) directions, sigma=0.2, 16 complete trajectories per plus/minus perturbation, '
    'learning rate 0.1 and update L2 cap 0.2. The estimator is `mean((J_plus-J_minus)*epsilon/(2*sigma))`; the update subtracts it. '
    'It includes discrete sampling and geometry through full-return evaluation and estimates a Gaussian-smoothed parameter objective; '
    'no pathwise derivative through a hard reward is assumed. The update cap is not another loss.','',
    'Directions use weight columns 2:6 from the original seeded six-dimensional Gaussian draws. Their marginal law remains iid N(0,I4). '
    'Each plus/minus pair uses the old trajectory key `610000 + seed*10000 + iteration*100 + direction`; '
    'the matched six-parameter run uses the same keys and weight-coordinate perturbations. Removing two dimensions changes perturbation geometry '
    'and norm clipping, so this does not isolate every optimization effect.','',
    'Each new arm spends 1,536 parameter-query paths plus 208 fixed-monitor paths, exactly the old per-arm budget. '
    'The final 12th update is evaluated; no monitor or evaluation outcome selects the checkpoint or changes settings. '
    'The original zero control and six-parameter search results are reused rather than retrained. Before reuse, all 512 reset paths and '
    'all 256 continuation paths per old arm reproduce exactly from the saved checkpoints. Their full action probes also reproduce exactly. '
    'The prior zero-control parameter-query budget was the same; no extra zero-control optimization is necessary here.','',
    'Evaluation reuses keys 710001..710004, 128 complete reset paths each. These keys are independent of search perturbations, queries and monitoring; '
    'they were already inspected in preceding experiments and are not a newly untouched test set. Bootstrap intervals pair the 512 simulated paths '
    'under common randomness and capture model Monte Carlo variation only, not causal or environment uncertainty.','',
    '## Model return and reward timing','']
  for name,label in zip(NAMES,LABELS):
    r=evaluation['reset'][name];e=extra[name]
    lines.append(f'- {label}: mean return {r["model_return_mean"]:.5f}; nonzero {r["positive_return_fraction"]:.2%}; '
      f'mean rewarded steps {r["mean_rewarded_steps"]:.2f}; first rewarded step among reaching paths {e["first_reward_step_conditional_on_any_reward"]:.2f}.')
  for seed in (0,1):
    a=evaluation['paired_four_minus_control'][f'four_s{seed}'];b=evaluation['paired_four_minus_six'][f'four_s{seed}']
    lines.append(f'- Seed {seed}, four minus control: {a["mean"]:.5f}, 95% interval {np.round(a["ci95_paired_model_paths"],5).tolist()}; '
      f'four minus six: {b["mean"]:.5f}, interval {np.round(b["ci95_paired_model_paths"],5).tolist()}.')
  lines+=['','Seed 0 retains a return reduction close to the six-parameter result; seed 1 loses much of the prior reduction. '
    'Removing biases does not eliminate the ability of the remaining weights to lower predicted return. These are model predictions, not actual environment returns.','',
    '![Return and reward timing](returns_and_timing.png)','',
    '## Direction, magnitude and geometry','',
    'Reuse the exact 32 predefined observed contexts (eight each: open corridor, boundary, pre-swamp, stationary non-reset), '
    'two saved nominal x_prime draws per context and 64 anchor draws per action. Keep s, goal, x_prime and key 950001 fixed while varying '
    'zero, opposing cardinal actions at magnitudes 0.5 and 1, and x=x_prime. Contexts are not reselected for this ablation; '
    'stationarity is not labeled death. Prescribed actions require no clipping.','',
    'Common-direction energy is `1 - sum ||d_a - mean_a d_a||^2 / sum ||d_a||^2` at each fixed context/anchor, '
    'aggregated across those groups and excluding zero-radius actions. d_a is the residual direction before multiplication by '
    '`0.25 * ||x-x_prime||`. A common direction can therefore have a nonzero action response purely through radius changes.','']
  for name,label in zip(NAMES[1:],LABELS[1:]):
    m=metrics[name]['all'];e=extra[name]
    lines.append(f'- {label}: common direction {m["common_direction_energy_fraction"]:.2%}; raw/emitted residual magnitude '
      f'{m["mean_raw_residual_magnitude"]:.5f}/{m["mean_emitted_anchor_displacement"]:.5f}; '
      f'opposing-X/Y response {m["half_opposing_x_response_norm"]:.5f}/{m["half_opposing_y_response_norm"]:.5f}; '
      f'projection/fallback {m["candidate_corrected_fraction"]:.3%}/{m["bound_fallback_fraction"]:.3%}; '
      f'action-specific normalized-direction RMS {e["direction_action_specific_rms"]:.5f}.')
  lines+=['','The drop from about 99.6-99.8% to about 96% common direction is not a sufficient improvement: the common component still dominates, '
    'and both raw residual and opposing-action response magnitudes shrink. The residual does not vanish entirely. '
    'The four-weight seed-0 mean normalized direction is left/down; seed 1 is left/up. Context-dependent random-feature weights '
    'can retain common translations without output biases. This is not evidence that the full MLP is incapable of meaningful action dependence.','',
    f'There is some increased action dependence in normalized direction: action-specific direction RMS rises from '
    f'{extra["six_s0"]["direction_action_specific_rms"]:.4f}/{extra["six_s1"]["direction_action_specific_rms"]:.4f} to '
    f'{extra["four_s0"]["direction_action_specific_rms"]:.4f}/{extra["four_s1"]["direction_action_specific_rms"]:.4f}. '
    'Thus this is not merely uniform scaling of the old direction field. Nevertheless, the common component remains dominant and the emitted opposing-action '
    'responses are weaker. Direction RMS across stochastic anchor samples is only '
    f'{extra["four_s0"]["direction_variation_across_anchor_samples_rms"]:.4f}/{extra["four_s1"]["direction_variation_across_anchor_samples_rms"]:.4f}; '
    'the change is not dominated by new anchor-dependent directional noise. Only '
    f'{extra["four_s0"]["raw_residual_near_zero_fraction"]:.2%}/{extra["four_s1"]["raw_residual_near_zero_fraction"]:.2%} '
    'of raw probe residuals have magnitude below 0.001. Complete disappearance is not the explanation either.','',
    '![Magnitude and action responses](action_response_comparison.png)','',
    'Raw and emitted samples remain distinct in the saved probes. Four-weight seed 0 now has more one-step geometry corrections than its six-parameter '
    'counterpart; those corrections do not constitute a successful action model. Inspecting only the common-direction fraction would miss this change. '
    'Control output remains exactly action-invariant under fixed randomness.','',
    '![Same representative contexts, raw and emitted](raw_and_emitted_examples.png)','',
    '## Stationary and moving history continuations','',
    'Use the original 32 distinct-episode stationary non-reset contexts and 32 moving contexts, key 720001 and four paths per context. '
    'The actor samples x and nominal samples x_prime at each generated state; the commanded goal stays fixed. '
    'Only `50 - original timestep` transitions contribute; later padded records are inactive, and history checks count active steps only. '
    'These are observed-data starts followed by model-generated continuations, not simulator ground truth.','']
  for name,label in zip(NAMES,LABELS):
    stationary=evaluation['continuation'][name]['observed_frozen_nonreset'];moving=evaluation['continuation'][name]['observed_moving']
    lines.append(f'- {label}: stationary-history first-step motion {stationary["mean_first_step_motion"]:.5f}, '
      f'remaining return {stationary["model_return_mean"]:.5f}; moving-history first-step motion {moving["mean_first_step_motion"]:.5f}, '
      f'remaining return {moving["model_return_mean"]:.5f}.')
  lines+=['','Removing biases moderates the increased movement from stationary histories, but both new arms still move those histories more than the control. '
    'No stationary history is silently forced to remain stationary and no hidden death labels are used. Exact F4 shift does not validate '
    'multi-step causal prediction or guarantee physical reachability.','',
    '![Continuation behavior](continuation_comparison.png)','',
    '## Interpretation and recommendation','',
    'The biases contributed to the strength of the previous drift and, for seed 1, much of the return reduction. They are not necessary for the '
    'common-direction shortcut: the remaining four weights still produce it in both seeds. A slightly lower common-direction fraction alongside '
    'weaker opposing-action responses is not evidence of learning action consequences. The effect neither disappears entirely nor becomes a '
    'well-supported action-dependent pessimistic mechanism. Dimension-dependent search geometry prevents a precise causal attribution of effect size to biases alone.','',
    '**Do not continue repeated parameter tuning in this four-/six-coordinate search family.** It remains useful as a diagnostic of how the hard-return '
    'objective can be reduced, but the present evidence does not justify using it to learn transitions for alternating contrastive-agent training. '
    'Any later work would need separately motivated execution-action structure or evidence, rather than another setting chosen to lower the same model return. '
    'No new architecture, action bound, third loss, actor-critic training or simulator-fitting requirement was introduced. '
    'Individual paired simulator outcomes from the preceding report are not exact conditional ETT ground truth and are not an equality target here. '
    'No causal validity or strict worst-case Q recovery is claimed.','',
    '## Completed checks and reproducibility','',
    f'Five numerical tests and a separate one-update smoke passed. The full run took {config["elapsed_seconds"]:.1f}s on {config["runtime"]["devices"]}. '
    'Saved checkpoint verification independently reloads the original initialization, checks only rows 0/32 changed, checks exactly zero biases '
    'and unchanged diagonal likelihood/parameters, confirms exact x=x_prime sampling, and reproduces 128 saved reset paths per new checkpoint. '
    'Objective records, goal/action bounds, F4 shifts, remaining horizons and raw-versus-float64 reward indicators are checked in `verification.json`. '
    'All old input/checkpoint hashes remain unchanged. Code, configurations and evaluation artifacts are versioned; checkpoints remain local.','',
    '```bash','python -m scripts.test_bias_ablation',config['command'],
    f'python -m scripts.check_bias_ablation --run-dir {out.as_posix()}',
    f'python -m ett.report_bias_ablation --run-dir {out.as_posix()}','```','',
    'Use a fresh output directory. `config.json` records paths, input hashes, seeds, source hashes and the fixed budget. '
    '`optimization_history.json` records all signed query returns, gradients and four-coordinate updates; perturbation arrays are saved separately. '
    '`evaluation.json` maps every reused/new artifact to its source. `four_s0.pkl` and `four_s1.pkl` stay local and Git-ignored. '
    'New rollout files separate agent actions from auxiliary observational actions.']
  (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
  write_json(out/'report_provenance.json',{'source_sha256':file_sha(__file__),'uses_saved_results_only':True,
      'command':f'python -m ett.report_bias_ablation --run-dir {out.as_posix()}'})
  print('wrote bias-ablation report and four figures',flush=True)


if __name__=='__main__':main()
