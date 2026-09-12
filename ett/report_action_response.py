"""Render saved action probes; no model sampling, optimization or training."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from ett.action_response import ACTION_NAMES,GROUPS
from ett.diagonal_transition import POINTMAZE_WALLS
from ett.run_rollout_return import read
from ett.run_distribution_matching import write_json
from scripts.make_swamp_f4_failure_bank import file_sha


NAMES=('control','return_seed0','return_seed1')
LABELS=('Frozen control','Return search seed 0','Return search seed 1')
COLORS=('#657580','#ba5329','#315fb0')
SHORT=('Zero','Right .5','Left .5','Up .5','Down .5','Right 1','Left 1','Up 1','Down 1','x = x_prime')


def load(path):
  with np.load(path,allow_pickle=False) as data:return {k:data[k] for k in data.files}


def figures(out,records,teacher,reference,groups,sim_groups,continuation):
  colors=plt.get_cmap('tab10')(np.arange(10))
  fig,axes=plt.subplots(4,3,figsize=(12,12),constrained_layout=True)
  selected=[]
  for row,group in enumerate(GROUPS):
    index=int(np.flatnonzero(groups==group)[0]);selected.append(index)
    for col,(name,label) in enumerate(zip(NAMES,LABELS)):
      ax=axes[row,col];record=records[name]
      displacement=(record['next_state'][index,:,:,:2]-record['anchor_xy'][index]).mean(1)
      raw=(record['proposal_xy'][index]-record['anchor_xy'][index]).mean(1)
      for a in range(10):
        ax.plot([0,raw[a,0]],[0,raw[a,1]],':',color=colors[a],alpha=.6)
        ax.annotate('',xy=displacement[a],xytext=(0,0),arrowprops={'arrowstyle':'->','color':colors[a],'lw':1.2})
        ax.plot(*raw[a],'x',color=colors[a],ms=5)
      ax.plot(0,0,'k*',ms=7)
      ax.axhline(0,c='.8',lw=.6);ax.axvline(0,c='.8',lw=.6)
      ax.set(xlim=(-.32,.12),ylim=(-.15,.3),aspect='equal',xlabel='X change from sampled anchor',ylabel='Y change from sampled anchor')
      ax.set_title(f'{label}\n{group}; state XY={record["state"][index,:2].round(2)}',fontsize=9)
  handles=[plt.Line2D([0],[0],color=colors[a],label=SHORT[a]) for a in range(10)]
  fig.legend(handles=handles,loc='outside lower center',ncol=5,fontsize=8)
  fig.suptitle('First selected context per group, first nominal draw: paired mean action responses\nStar = anchor; arrows = emitted displacement; dotted lines / crosses = raw proposals',fontsize=12)
  fig.savefig(out/'action_responses.png',dpi=150);plt.close(fig)
  # Explicitly select the largest projection/fallback event, not a typical event.
  r=records['return_seed1'];distance=np.linalg.norm(r['next_state'][...,:2]-r['proposal_xy'],axis=-1)
  context,action,sample=map(int,np.unravel_index(distance.argmax(),distance.shape))
  fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
  ax=axes[0];ax.imshow(POINTMAZE_WALLS.T,origin='lower',extent=(0,9,0,5),cmap='Greys',alpha=.3,vmin=0,vmax=1)
  points=[('Current state',r['state'][context,:2],'o','#333333'),
          ('Diagonal anchor',r['anchor_xy'][context,action,sample],'o','#315fb0'),
          ('Raw proposal',r['proposal_xy'][context,action,sample],'x','#ba5329'),
          ('Projected candidate',r['projected_xy'][context,action,sample],'s','#7b4591'),
          ('Emitted output',r['next_state'][context,action,sample,:2],'*','#ce3328')]
  for label,point,marker,color in points:
    options={'edgecolors':color,'facecolors':'none'} if marker=='s' else {'color':color}
    ax.scatter(*point,marker=marker,s=110 if marker=='*' else 60,label=label,**options)
  path=np.array([v[1] for v in points[1:]])
  ax.plot(path[:,0],path[:,1],'--',c='gray',lw=1)
  lo=np.min(np.array([v[1] for v in points]),axis=0)-.25;hi=np.max(np.array([v[1] for v in points]),axis=0)+.25
  ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),aspect='equal',title=f'Largest seed-1 geometry event: {SHORT[action]}',xlabel='X',ylabel='Y');ax.legend(fontsize=8)
  for field,label,color in [('proposal_xy','Raw proposal','#ba5329'),('projected_xy','After geometry','#7b4591'),('next_state','Emitted','#315fb0')]:
    displacement=r[field][context,:,sample,:2]-r['anchor_xy'][context,:,sample]
    axes[1].plot(np.arange(10),np.linalg.norm(displacement,axis=-1),'o-',label=label,c=color)
  axes[1].plot(np.arange(10),r['radius'][context,:,sample],'k--',label='Original anchor bound')
  axes[1].set_xticks(np.arange(10),SHORT,rotation=55,ha='right');axes[1].set(ylabel='Distance from same sampled anchor',title='Same state, x_prime and sample; only execution action varies');axes[1].legend(fontsize=8)
  fig.savefig(out/'projection_effect.png',dpi=160);plt.close(fig)
  example={'selection':'maximum emitted-versus-raw proposal difference in seed-1 offline probes',
      'context_times_nominal_draw_index':context,'action_index':action,'sample_index':sample,
      'difference':float(distance[context,action,sample]),'bound_fallback':bool(r['bound_fallback'][context,action,sample]),
      'points':{label:point for label,point,_,_ in points}}
  fig,axes=plt.subplots(4,2,figsize=(12,11),constrained_layout=True)
  sim_selected=[]
  for row,group in enumerate(GROUPS):
    indices=np.flatnonzero(sim_groups==group)
    if not len(indices):
      for ax in axes[row]:ax.set_visible(False)
      continue
    c=int(indices[0]);sim_selected.append(c)
    true=reference['states'][c,:,1,:2]-reference['states'][c,-1,1,:2]
    for dimension,ax in enumerate(axes[row]):
      ax.plot(np.arange(10),true[:,dimension],'ko-',lw=1.5,label='One paired simulator context')
      for name,label,color in zip(NAMES,LABELS,COLORS):
        pred=teacher[name]['next_state'][c,:,:,:2].mean(1)
        ax.plot(np.arange(10),(pred-pred[-1])[...,dimension],'o-',ms=3,label=label,c=color)
      ax.set_xticks(np.arange(10),SHORT,rotation=55,ha='right',fontsize=7)
      ax.set_title(f'{group}: {"XY"[dimension]} response relative to x = x_prime',fontsize=9)
      ax.axhline(0,c='.7',lw=.6);ax.set_ylabel('Maze units')
      if row==0:ax.legend(fontsize=7)
  fig.suptitle('Natural teacher x_prime paired to its own hidden context; first selected example per group\nModel means versus a single paired outcome, not conditional ETT ground truth',fontsize=12)
  fig.savefig(out/'teacher_paired_responses.png',dpi=150);plt.close(fig)
  fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
  for name,label,color in zip(NAMES,LABELS,COLORS):
    data=continuation[name]
    axes[0].plot(range(6),data['mean_position_discrepancy_by_step'],'o-',label=label,c=color)
    axes[1].plot(range(6),data['opposing_action_contrast_discrepancy_by_step'],'o-',label=label,c=color)
    axes[2].plot(range(1,6),np.array(data['candidate_corrected_by_step'])*100,'o-',label=label,c=color)
  axes[0].set(xlabel='Prescribed-action step',ylabel='Mean XY distance',title='Disagreement with paired simulator positions')
  axes[1].set(xlabel='Prescribed-action step',ylabel='Mean response-vector distance',title='Right .5 versus left .5 contrast disagreement')
  axes[2].set(xlabel='Prescribed-action step',ylabel='Corrected model samples (%)',title='Geometry correction frequency')
  for ax in axes:ax.legend(fontsize=7)
  fig.suptitle('Five-step prescribed comparisons: no actor, no hindsight goal, no conditional oracle claim',fontsize=12)
  fig.savefig(out/'short_continuations.png',dpi=150);plt.close(fig)
  return {'offline_examples':selected,'teacher_examples':sim_selected,'projection_example':example}


def main():
  p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',required=True);args=p.parse_args();out=Path(args.run_dir)
  config=read(out/'config.json')
  if config['status']!='complete':raise ValueError('diagnostic not complete')
  contexts=read(out/'contexts.json');audit=read(out/'simulator_restoration_audit.json')
  continuation_contexts=read(out/'continuation_contexts.json')
  groups=np.repeat(contexts['groups'],2);sim_groups=np.array(audit['groups'])
  metrics=read(out/'model_probe_metrics.json');sim=read(out/'simulator_comparison.json');continuation=read(out/'continuation_metrics.json')
  records={name:load(out/f'{name}_model_probes.npz') for name in NAMES}
  teacher={name:load(out/f'{name}_teacher_probes.npz') for name in NAMES}
  reference=load(out/'simulator_one_step.npz');protocol=read(out/'simulator_protocol.json')
  examples=figures(out,records,teacher,reference,groups,sim_groups,continuation)
  write_json(out/'example_selection.json',examples)
  actual=reference['states'][:,:,1,:2];xp=reference['observational_action'];actions=reference['execution_action']
  reference_change=np.linalg.norm(actual-actual[:,-1:],axis=-1)
  radius=.25*np.linalg.norm(actions-xp[:,None],axis=-1)
  constraints={'paired_simulator_change_exceeds_model_coupling_radius_fraction':float((reference_change>radius+1e-6).mean()),
      'scope':'comparison to the same-underlying-state simulator coupling, not a model constraint violation and not a required equality for pessimism'}
  write_json(out/'reference_coupling_limits.json',constraints)
  checks=read(out/'numerical_checks.json');max_error=max(v['maximum_reconstruction_error'] for v in checks.values())
  lines=['# PointMaze ETT execution-action response diagnostic','',
    '**Diagnosis: immediate action-semantics limitations and mostly common residual drift, with occasional geometry artifacts and compounding model disagreement.** '
    'The zero-residual control is exactly action-invariant in these fixed-randomness probes. Both searched models respond to x, '
    'but mainly by scaling a nearly common direction with distance from x_prime. Projection/fallback is secondary in the one-step sample. '
    'Do not start alternating actor-critic training on this evidence.','',
    '## Preserved scope and provenance','',
    'Reviewed commit and verified local/remote branch head: `03f7686a2277c684d3801a6dcb7f24bd81ea622c`, '
    '`feature/pointmaze-causal-transition`. No applicable repository AGENTS.md was found. '
    'The rollout-return, preceding MMD and nearest-set reports were read. The critic report covering `e670a33`, `6947186`, and `089f083` '
    'was read and its rejected raw score was neither recomputed nor substituted.','',
    'The objective remains exactly `L_diag + lambda * L_off`. No loss or optimization gradient is computed in this diagnostic, '
    'and no model, nominal, actor or critic is trained. The preceding return search froze the diagonal backbone and changed only six residual-head '
    'coordinates; it was not joint two-loss training. No architecture or geometry rule is changed here.','',
    'Checkpoints come from `artifacts/ett_rollout_return/residual6_s01`: `s0_lambda0.pkl` (control), '
    '`s0_lambda1.pkl` (return seed 0), and `s1_lambda1.pkl` (return seed 1). Their hashes match the saved return-search evaluation. '
    'Exact checkpoint, nominal and dataset paths and SHA-256 hashes are in `config.json`. All prior input artifacts are unchanged.','',
    'This `f4_return_s01_horizon50` run supersedes the preliminary `f4_return_s01` continuation evaluation. '
    'The preliminary one-step probes remain valid, but its timestep-48 context exceeded the task horizon in a five-step continuation. '
    'The final run applies the explicit remaining-horizon eligibility rule; old output files remain intact.','',
    'Contract: fixed-goal `point_two_route_swamp_windy_f4_v0`, p=0.30; newest-first F4 width 8; goal `(8.5,3.5)` repeated four times; '
    'action width 2 with bounds [-1,1]. Nominal conditioning remains state plus commanded goal. '
    'Horizon is 50 with no success/death early termination; this task limits prescribed continuations to five steps. '
    'No rewards are fitted or used as targets, and no hidden value enters model inputs.','',
    'Population clarification from the actual loader: the nominal fits episodes [1200,6000), generated by '
    '`make_windy_teacher`. Uniform coverage [0,1200) and blind demonstrations [6000,6600) are excluded. '
    'The collector-wide random_frac=0.2 does NOT mean 20% uniform episodes remain in this selected population. '
    'Within the selected 4,800 episodes, mode counts are forced-safe 257, immediate-shortcut 1,470, and wait-shortcut 3,073. '
    'The teacher uses a 0.05 force-safe episode coin and adds N(0,0.15^2 I) noise only to nonzero commands, then clips. '
    'This is noisy teacher behavior including failures, not uniformly optimal actions.','',
    '## Predefined probes and instrumentation','',
    f'Selected {len(contexts["groups"])} held-out source contexts: eight per stratum, using seed 940000, '
    'a row permutation and distinct source episodes within each stratum. Groups may share source episodes. '
    'Rules: open post-swamp corridor x in [6.2,8.2], y in (3.2,3.8); actual static-wall/boundary distance <=0.06; '
    'pre-swamp x in [2,3), y in [3,4); and fully equal observed F4 frames at non-reset timesteps. '
    'The first three strata exclude fully stationary histories. Stillness is not labeled death. Exact rows, episode IDs, histories, '
    'goals, eligible counts and rules are saved in `contexts.json`.','',
    f'Two nominal x_prime draws per context (seed 950000), {config["arguments"]["samples"]} transition draws per action '
    '(key 950001), ten actions: zero, four cardinal directions at magnitudes 0.5 and 1, and x=x_prime. '
    'Each action call has identical batch ordering, s, g, x_prime and transition key. All model arms share those inputs and keys. '
    'Prescribed actions need no clipping; out-of-bound x_prime is rejected. Boundary mass in nominal draws is its existing censored distribution.','',
    'The original `AnchoredTransition.sample` implementation is untouched. Instrumentation first calls it, then reconstructs raw network outputs, '
    'tanh directions, safe radii, raw proposals and projection/fallback stages. A repeated ordinary sampler call is bit-identical. '
    f'The largest intermediate reconstruction difference is {max_error:.3g} maze units (float32 JIT fusion tolerance 2e-6). '
    'Returned samples always come from the original implementation. Raw logit proposals, projected positions, emitted F4 states and corrections are retained.','',
    'All probes pass exact F4 shift, exact x=x_prime anchor identity, common anchor across actions, endpoint, per-coordinate displacement and anchor-bound checks. '
    'These explicit constraints are not full physics or causal-validity guarantees.','',
    '## Observations: action dependence versus common drift','',
    'The control has maximum execution-action effect **0.0** under fixed randomness. Its anchor is sampled at `(s,x_prime,x_prime)`; '
    'x is absent from the anchor distribution. At zero residual the emitted sample equals that anchor for every tested x.','',
    'For each fixed context/nominal draw/anchor sample, let d_a be the normalized residual direction before radius multiplication. '
    'The common-direction energy fraction is `1 - sum ||d_a - mean_a(d_a)||^2 / sum ||d_a||^2`, excluding zero-radius actions. '
    'It measures direction constancy across actions within a context, not constancy of residual magnitude or independence from context. '
    'A direction independent of action can still produce action sensitivity because the radius is `0.25 * ||x-x_prime||`.','']
  for name in NAMES[1:]:
    m=metrics[name]['all']
    lines.append(f'- {name}: common-direction energy {m["common_direction_energy_fraction"]:.3%}; '
      f'mean normalized direction {np.round(m["mean_common_direction"],4).tolist()}; '
      f'paired +/-0.5 X response magnitude {m["half_opposing_x_response_norm"]:.5f}, Y response {m["half_opposing_y_response_norm"]:.5f}; '
      f'candidate correction {m["candidate_corrected_fraction"]:.3%}; bound fallback {m["bound_fallback_fraction"]:.3%}.')
  lines+=['','Directions vary somewhat with context: seed 0 common-direction fractions span '
    f'{min(metrics["return_seed0"][g]["common_direction_energy_fraction"] for g in GROUPS):.2%} to '
    f'{max(metrics["return_seed0"][g]["common_direction_energy_fraction"] for g in GROUPS):.2%}. '
    'The dominant direction remains leftward for seed 0 and left/up for seed 1 across all four groups. '
    'Nonzero action derivatives alone would therefore be an inadequate success criterion.','',
    '![Paired model responses](action_responses.png)','',
    'Seed 0 has no material one-step geometry correction in this selected sample; its drift precedes geometry. '
    'Seed 1 has rare corrections in pre-swamp contexts. The ratio of geometry-response variance to emitted-action variance is '
    f'{metrics["return_seed1"]["all"]["geometry_to_emitted_action_variance_ratio"]:.4f}. '
    'This is not an additive attribution fraction because raw response and correction have a cross term. '
    'The selected worst geometry example proposes a point across a wall; endpoint projection returns to the current position, '
    'which would exceed the anchor radius, so final fallback returns to the anchor. The discontinuity is present but does not explain most one-step drift.','',
    '![Projection and fallback example](projection_effect.png)','',
    'These are checkpoint-specific observations. The full residual MLP can depend on x; the result does not prove that every possible parameter '
    'setting is action-invariant or that the entire class lacks expressive capacity. The six-dimensional search and action-independent output biases are material restrictions.','',
    '## Paired simulator reference and its conditioning limits','',
    f'Generated {protocol["episodes"]} fresh full teacher episodes ({protocol["natural_steps"]} natural transitions) with environment seed 930000 and behavior seed 930001. '
    f'Teacher mode counts: {protocol["mode_counts"]}; noisy-command clipping occurred on {protocol["teacher_noise_clipped_steps"]} steps. '
    'The same imported teacher, episode coin, action-noise rule, environment probability and history contract define the selected nominal source population. '
    'Fresh PRNG streams do not reproduce original dataset episodes, but use the same generating protocol. This does not prove that the fitted nominal density is exact.','',
    f'Selected {len(sim_groups)} simulator contexts by the same observable rules with selection seed 940001: four each for open, boundary and pre-swamp, '
    'and only three stationary non-reset contexts from distinct episodes. The fixed 64-episode pool contains only three eligible stationary episodes; '
    'the budget was not enlarged to fill the quota.','',
    'At every selected context, the natural noisy teacher action x_prime is generated before any intervention and remains attached to that context. '
    'Full float64 physical position, F4 frames, goal, hidden bits, absorbing flag, environment generator state, teacher generator state and teacher memo '
    'are saved in a separately labeled restoration audit. Replaying the teacher action and natural simulator transition reproduces action, observation, '
    'reward and post-step RNG state exactly. The model sees only float32 F4, goal and the two actions.','',
    'Alternative actions execute on clones of that same pre-action underlying context. The first step shares the native generator state. '
    'For later steps, normal action-noise and uniform bit-resampling slots are independently preallocated by timestep and shared across branches. '
    'This avoids RNG stream misalignment when an absorbing branch skips action noise. It preserves the simulator IID marginal noise law, '
    'while declaring a particular temporal cross-branch coupling; the coupling is not identified from offline observations.','',
    '**These are individual paired intervention outcomes, not exact conditional ETT distributions.** '
    'No independently sampled nominal x_prime is attached to an unrelated hidden simulator state. '
    'No continuous-action nearest matching, rejection approximation, or posterior averaging over all hidden contexts compatible with `(s,x_prime)` is performed. '
    'Averaging discrepancies over different selected contexts does not turn these pairs into an estimate of that conditional distribution.','',
    '![Teacher-paired action responses](teacher_paired_responses.png)','']
  for name in NAMES:
    lines.append(f'- {name}: mean one-step action-contrast discrepancy against the individual simulator pairs '
      f'{sim[name]["mean_action_contrast_discrepancy_to_single_paired_outcome"]:.5f} maze units (contrast relative to x=x_prime).')
  lines+=['','The simulator often responds strongly to alternative execution actions in moving open contexts, while modeled contrasts are much smaller. '
    'Near walls and with hidden dynamics, a forward action is not universally required to move forward. '
    'The three simulator stationary contexts have zero paired action contrasts; this is an observed result for those restored contexts, '
    'not a death label inferred from the offline stationary stratum.','',
    f'{constraints["paired_simulator_change_exceeds_model_coupling_radius_fraction"]:.1%} of these individual simulator action pairs exceed '
    'the model family\'s 0.25 times action-distance radius when measured around their own natural-action simulator outcome. '
    'That radius was never verified as a physical causal constant. This exposes a limitation of demanding equality under this specific coupling; '
    'it is neither a violation by the constrained model nor a reason to require every pessimistic transition to equal the true simulator.','',
    '## Five-step prescribed continuations','',
    f'The {len(continuation_contexts["one_step_context_indices"])} selected simulator contexts with at least five steps remaining have '
    'three fixed action sequences: zero, right 0.5, left 0.5, repeated for five steps. '
    'The one-step context at t=48 is excluded by the original 50-step task horizon, not by its outcome. '
    'The commanded goal stays fixed. These are simulator-generated intervention references, not observed-data continuations or closed-loop actor comparisons. '
    'Model step 0 uses the natural teacher x_prime; subsequent model steps sample fresh nominal x_prime at their own generated visible states. '
    'No hidden label forces modeled stationary histories to stay still. There is one future simulator noise tape per context and 32 modeled paths per action/arm.','']
  for name in NAMES:
    c=continuation[name]
    lines.append(f'- {name}: mean positional discrepancy step 1 -> step 5: '
      f'{c["mean_position_discrepancy_by_step"][1]:.4f} -> {c["mean_position_discrepancy_by_step"][5]:.4f}; '
      f'opposing-action contrast discrepancy {c["opposing_action_contrast_discrepancy_by_step"][1]:.4f} -> '
      f'{c["opposing_action_contrast_discrepancy_by_step"][5]:.4f}; step-5 geometry corrections {c["candidate_corrected_by_step"][4]:.2%}.')
  lines+=['','![Short prescribed continuations](short_continuations.png)','',
    'Execution-action disagreement is already visible after one step; it is not caused only by repeated prediction or an F4 shift bug. '
    'Disagreement grows with repeated steps and geometry corrections become more frequent, especially for seed 1. '
    'Exact F4 shifting does not establish that four frames are a sufficient causal state, or distinguish transition bias from nominal misspecification '
    'under intervened state distributions. Single-context comparisons cannot resolve those questions.','',
    '## Interpretation, smallest next change, and limits','',
    'Evidence supports a combination: the control has no execution-action semantics; the fitted residual largely adds directional drift with '
    'action-distance-dependent magnitude; rare projection/fallback discontinuities occur; repeated model predictions amplify disagreement. '
    'There is no observed history-order or explicit output-bound violation. Simulator disagreement and unsupported changes in dynamics are distinct '
    'from violations of the implemented structural constraints. No new admissibility rule was invented to reject lower-return outcomes.','',
    '**Smallest justified next ablation:** freeze the two residual output-bias coordinates at zero and, if separately authorized, repeat the same '
    'small search using only the four previously selected output-weight coordinates. Keep the same initialization, two losses, sampling budget and '
    'geometry constraints, and evaluate action contrasts before interpreting return reduction. This tests whether the unrestricted output-bias directions '
    'are a principal route to common drift. It introduces a parameter-subspace restriction, not a verified physical law; the remaining random-feature '
    'weights may still produce common drift, and the zero-residual control still ignores x. It is a diagnostic ablation, not a claimed fix for causal ETT. '
    'No such ablation, new architecture, return optimization, critic training or alternation was launched here.','',
    'Do not connect this model to contrastive-agent alternating training yet. Neither exact simulator agreement nor strict worst-case Q is established '
    'or required by this diagnostic. What remains unsupported is the interpretation of the learned low-return transition as a useful action-conditioned '
    'pessimistic mechanism.','',
    '## Artifacts and reproduction','',
    f'Completed six checkpoint-independent numerical tests, a small smoke, and this bounded run in {config["elapsed_seconds"]:.1f}s on '
    f'{config["runtime"]["devices"]}. Contexts, seeds, hashes, eligibility counts, native restoration checks, raw proposals, anchors, corrections, '
    'emitted states and representative examples are saved. All existing checkpoints and artifacts are unchanged. No training or model updates were performed.','',
    '`verification.json` additionally confirms exact replay from the saved JSON restoration records, exact zero-control invariance, '
    'common anchors across models, finite saved arrays, history/geometry/action checks and original-horizon compliance.','',
    '```bash','python -m scripts.test_action_response',config['command'],
    f'python -m ett.report_action_response --run-dir {out.as_posix()}',
    f'python -m scripts.check_action_response_artifacts --run-dir {out.as_posix()}','```','',
    'Use a fresh output directory for reruns. `*_model_probes.npz` and `*_teacher_probes.npz` have leading axes '
    '`[context (including nominal-draw replication for offline probes), execution-action index, transition sample]`. '
    'Main continuation files contain model states; observational actions are stored in separate auxiliary files. '
    'The hidden simulator restoration/audit JSON files are evaluation-only and never consumed by the model.']
  (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
  write_json(out/'report_provenance.json',{'source_sha256':file_sha(__file__),'uses_saved_probes_only':True,
      'command':f'python -m ett.report_action_response --run-dir {out.as_posix()}'})
  print('wrote action-response report and four figures',flush=True)


if __name__=='__main__':main()
