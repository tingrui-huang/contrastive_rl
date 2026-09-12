"""Report saved hard-return rollouts without new fitting or model draws."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle,Rectangle
import numpy as np

from ett.diagonal_transition import POINTMAZE_WALLS
from ett.rollout_return import GOAL,DISCOUNT,task_reward
from ett.run_rollout_return import read
from ett.run_distribution_matching import write_json
from scripts.make_swamp_f4_failure_bank import file_sha

NAMES=('s0_lambda0','s0_lambda1','s1_lambda1')
LABELS=('Frozen control','Return search seed 0','Return search seed 1')
COLORS=('#607d8b','#c25729','#315ab2')


def load_records(out):
  records={}
  for name in NAMES:
    with np.load(out/f'{name}_trajectories.npz',allow_pickle=False) as data:
      records[name]={k:data[k] for k in data.files}
  return records


def describe_paths(records):
  stats={}
  for name,data in records.items():
    rewards=data['reward'][0];xy=data['states'][0,...,:2]
    hit=(rewards>0).any(-1)
    first=np.where(hit,np.argmax(rewards>0,axis=-1)+1,51)
    movement=np.linalg.norm(np.diff(xy,axis=1),axis=-1)
    stats[name]={'first_reward_step_if_reached':float(first[hit].mean()) if hit.any() else None,
        'mean_path_length':float(movement.sum(1).mean()),
        'last_10_steps_near_stationary_fraction':float((movement[:,-10:]<=.05).mean()),
        'exactly_stationary_last_10_steps_path_fraction':float((movement[:,-10:]==0).all(-1).mean()),
        'reward_delay_censored_at_51_mean':float(first.mean())}
  return stats


def reward_precision_audit(out):
  """Separate observation quantization from the task's hidden death gate.

  Recompute saved rewards only; three deterministic boundary probes expose
  information lost when the simulator casts its internal position to float32.
  These probes are evaluation evidence and never become optimization inputs.
  """
  from crl.envs import TwoRouteSwampWindyF4Env
  probes=[]
  for x in (6.5,6.5000001,6.500001):
    env=TwoRouteSwampWindyF4Env(action_noise=0,seed=2)
    env.state=np.array([x,3.5],np.float64)
    observation,reward,_,_=env.step(np.zeros(2,np.float32))
    probes.append({'initialized_x_float64':x,'visible_xy_float32':observation[:2],
        'environment_reward':reward,'visible_reward':float(task_reward(observation[:8],observation[8:]))})
  saved={}
  for path in sorted(out.glob('*_trajectories.npz')):
    with np.load(path,allow_pickle=False) as data:
      xy=data['states'][...,1:,:2];goal=data['commanded_goal'][...,None,:2]
      active=data['active'];recorded=data['reward']
      distance64=np.linalg.norm(xy.astype(np.float64)-goal.astype(np.float64),axis=-1)
      reward32=np.asarray(task_reward(xy,goal))*active
      reward64=(distance64<2.)*active
      if not np.array_equal(recorded,reward32):raise RuntimeError(f'saved reward mismatch: {path}')
      discount=np.power(DISCOUNT,np.arange(recorded.shape[-1]))
      saved[path.name]={'active_transitions':int(active.sum()),
          'recorded_vs_float32_reward_mismatches':int(np.count_nonzero(recorded!=reward32)),
          'float32_vs_float64_predicate_mismatches':int(np.count_nonzero((reward32!=reward64)&active)),
          'distance_within_1e_minus_6_of_boundary':int(np.count_nonzero((np.abs(distance64-2.)<=1e-6)&active)),
          'maximum_path_return_difference_float64_predicate':float(np.abs(((reward64-reward32)*discount).sum(-1)).max())}
  result={'scope':'post-run numerical audit only; no reward change or retraining',
      'model_reward_convention':'strict radius-2 hard indicator on the emitted float32 visible position, evaluated with float32 arithmetic',
      'universal_bit_exact_environment_reconstruction_from_float32_observations':False,
      'reason':'float64 internal positions on opposite sides of the strict boundary can map to the same float32 observation; the hidden death gate is separately disjoint from this fixed reward region',
      'boundary_probes':probes,'saved_trajectory_checks':saved}
  write_json(out/'reward_precision_audit.json',result)
  return result


def figures(out,records,evaluation,history):
  fig,axes=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
  for seed,color in zip((0,1),COLORS[1:]):
    for weight in (0,1):
      h=history[f's{seed}_lambda{weight}']
      axes[0].plot([r['iteration'] for r in h],[r['center_monitor_return'] for r in h],
          label=f'seed {seed}, lambda {weight}',color=color,ls='-' if weight else '--')
  axes[0].set(xlabel='Fixed-budget search iteration',ylabel='Discounted hard return',title='Training monitor, not used for selection')
  for name,label,color in zip(NAMES,LABELS,COLORS):
    r=records[name]['return'].ravel()
    axes[1].step(np.sort(r),np.arange(1,len(r)+1)/len(r),where='post',label=label,color=color)
    axes[2].plot(np.arange(1,51),records[name]['reward'].mean((0,1)),label=label,color=color)
  axes[1].set(xlabel='Predicted discounted return',ylabel='Empirical CDF',title='512 independent evaluation paths per arm')
  axes[2].set(xlabel='Step after reset',ylabel='Fraction receiving hard task reward',title='Goal radius 2; no success termination')
  for ax in axes:ax.legend(fontsize=8)
  fig.suptitle('Restricted ETT residual search: model-predicted returns only')
  fig.savefig(out/'returns_and_reward_timing.png',dpi=160);plt.close(fig)
  names=['stationary_fraction','mean_motion','near_wall_boundary_fraction','candidate_corrected_fraction']
  titles=['Exact stationarity','Mean XY step length','Near integer grid boundaries','Candidate geometry correction']
  fig,axes=plt.subplots(1,4,figsize=(14,3.5),constrained_layout=True)
  for ax,key,title in zip(axes,names,titles):
    ax.bar(np.arange(3),[evaluation['models'][n][key] for n in NAMES],color=COLORS)
    ax.set_xticks(np.arange(3),['Control','Seed 0','Seed 1'],rotation=25);ax.set_title(title,fontsize=10)
  fig.suptitle('Lower model return can accompany drift or boundary-related stagnation')
  fig.savefig(out/'motion_and_geometry.png',dpi=160);plt.close(fig)
  base=records['s0_lambda0']['return'][0]
  drop0=records['s0_lambda1']['return'][0]-base
  drop1=records['s1_lambda1']['return'][0]-base
  indices=[0,1,2,int(np.argmin(drop0)),int(np.argmin(drop1)),int(np.argmin(base))]
  selections=[]
  fig,axes=plt.subplots(2,3,figsize=(14,7),constrained_layout=True)
  for position,(ax,index) in enumerate(zip(axes.ravel(),indices)):
    ax.imshow(POINTMAZE_WALLS.T,origin='lower',extent=[0,9,0,5],cmap='Greys',vmin=0,vmax=1,alpha=.35)
    for x in (3,4,5):ax.add_patch(Rectangle((x,3),1,1,facecolor='#d4b453',alpha=.2))
    ax.add_patch(Circle(GOAL[:2],2,fill=False,ls='--',color='green'))
    entry={'path_index':index,'selection':['first path','second path','third path','largest seed-0 paired decrease','largest seed-1 paired decrease','lowest control return'][position]}
    for name,label,color in zip(NAMES,LABELS,COLORS):
      xy=records[name]['states'][0,index,:,:2]
      ret=float(records[name]['return'][0,index]);entry[name+'_return']=ret
      ax.plot(xy[:,0],xy[:,1],'.-',lw=1,ms=2,label=f'{label}: {ret:.2f}',color=color,alpha=.8)
      ax.plot(xy[-1,0],xy[-1,1],'x',color=color,ms=7)
    ax.set(xlim=(0,9),ylim=(0,5),aspect='equal',xlabel='X',ylabel='Y',title=f'Paired path {index}: {entry["selection"]}')
    ax.title.set_fontsize(9);ax.legend(fontsize=6,loc='lower center')
    selections.append(entry)
  fig.suptitle('Representative model rollouts under common randomness; crosses mark final positions\nStatic swamp geometry only; no hidden mask or generated-state failure labels')
  fig.savefig(out/'paired_trajectory_examples.png',dpi=160);plt.close(fig)
  write_json(out/'trajectory_examples.json',{'selection_scope':'first three and explicit return-extreme examples, qualitative not independent validation','examples':selections})
  fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
  for name,label,color in zip(NAMES,LABELS,COLORS):
    xy=records[name]['states'][0,...,:2]
    axes[0].plot(np.arange(51),xy[:,:,0].mean(0),color=color,label=label)
    axes[1].plot(np.arange(51),xy[:,:,1].mean(0),color=color,label=label)
  axes[0].set(xlabel='Model step',ylabel='Mean X',title='Progress and delayed arrival')
  axes[1].set(xlabel='Model step',ylabel='Mean Y',title='Directional drift');axes[1].axhline(4,color='gray',ls='--')
  for ax in axes:ax.legend(fontsize=8)
  fig.savefig(out/'mean_trajectory_positions.png',dpi=160);plt.close(fig)


def report(out,preflight,config,search,evaluation,continuation,paths,precision):
  lines=['# Bounded PointMaze hard-rollout-return experiment','',
      '**Decision: the second term is optimizable inside the model, but stop before joint or alternating actor-critic training.** '
      'Independent model returns decrease for both search seeds, while diagonal likelihood stays exactly fixed. '
      'The decrease accompanies drift, boundary-related stagnation and larger departures from observed frozen histories. '
      'It does not establish a physically valid counterfactual kernel or a worst-case Q function.','',
      '## Prior evidence and exact task contract','',
      'Branch HEAD at the start was `8eaee1370e1abeb1bf3df52ee693aca21221b93c`. '
      'The critic-continuation report covering commits `e670a33`, `6947186`, and `089f083` was read: '
      'the raw score is inverted on tight authentic dead/alive matches. That diagnostic was not repeated, '
      'and no critic or failure scorer was used. The preceding MMD and nearest-set reports were also read; '
      'the former improved dispersion rather than failure proximity, and the latter retained drift/collapse problems.','',
      'Task: `point_two_route_swamp_windy_f4_v0`, per-cell activation probability 0.30. '
      'The actual rollout horizon is 50, and the frozen actor training log confirms discount 0.95 '
      '(the generic config default of 0.99 is not used). The commanded goal is `(8.5,3.5)` tiled four times; '
      'reset is `(0.5,3.5)` tiled four times. State is newest-first F4, width 8; both actions have width 2 in [-1,1]. '
      'Both nominal and actor receive state plus the commanded goal, unchanged throughout every rollout.','',
      'The environment gives **one reward each step with newest-XY distance strictly below 2.0**, unless dead. '
      'It does not terminate on success or death; the caller stops after 50 transitions. '
      'The commonly reported reached@0.5 metric is only supplementary and is not the optimized reward. '
      'All absorbing deaths are located in swamp cells x in [3,6), y in [3,4), at least 2.5 units from this fixed goal. '
      'Thus the hidden death gate cannot change the mathematical position reward on reachable real states in this task. '
      'The rollout function explicitly rejects other goals. This position reward is not generalized to other tasks or arbitrary hidden-state assignments. '
      'Generated states need not be physically realizable.','',
      '**Finite-precision qualification:** the environment uses a float64 internal position and exposes float32 observations. '
      'Universal bit-exact reward reconstruction from those observations is therefore impossible at the strict radius boundary: '
      'internal x=6.5000001, y=3.5 is rewarded, but is observed as x=6.5, whose position indicator is zero. '
      'The model objective uses the strict radius-2 predicate on its emitted float32 state with float32 arithmetic; '
      'this is the stated numerical convention, not distance shaping. The 1,600 actual rollout rewards checked below agree, '
      'but do not establish equality for all boundary states. '
      f'Post-run precision checks found {sum(v["float32_vs_float64_predicate_mismatches"] for v in precision["saved_trajectory_checks"].values())} '
      'reward differences when evaluating the same saved model positions in float64, across all reset and continuation arms. '
      'The deterministic counterexamples and saved-sample checks are in `reward_precision_audit.json`; the training objective was not changed.','',
      'At each step, draw a fresh x_prime from the frozen goal-conditioned nominal, independently draw the agent action x, '
      'then sample T(s_next | s,x,x_prime,g). Every trajectory gets fresh policy and transition randomness; the goal stays fixed. '
      'Agent trajectory files store only x in `action`; x_prime and anchor diagnostics are in separate auxiliary files. '
      'There is no hindsight goal relabeling, hidden reward input, death label, bank target or new reward shaping.','',
      '## Preserved inputs and initial signal check','',
      'Dataset SHA-256: `83b4e81d9fca2d66b648c9c34ccdb68196e1acf89e2ca6829e4cb198d27c322c`. '
      'The unchanged expert-positive episode split has 4,321 training and 479 validation episodes '
      '(216,050 / 23,950 transitions). The source population is teacher-generated and is not uniformly optimal. '
      'The actor was originally fitted using all source episodes, so this is not actor-heldout evaluation. '
      'The policy loader still verifies source provenance; teacher modes never become rewards or model inputs.','',
      'Initialization: `ett_distribution_matching/f4_p30_s01_guarded/s0_L0p25_lambda0/final.pkl`. '
      'Its residual output head is exactly zero. The same frozen nominal K5 checkpoint and alpha-0 seed-0 actor are reused. '
      'Exact files and SHA-256 hashes are in `config.json`. The old MMD checkpoint is a preflight reference only; '
      'all search and control arms start from the identical diagonal-only checkpoint. No existing artifact was changed.','',
      f'Preflight completed 256 reset-to-horizon model paths each for control and old MMD, with shared keys 410001/410002. '
      f'Control return mean/std: {preflight["control"]["model_return_mean"]:.5f}/{preflight["control"]["model_return_std"]:.5f}; '
      f'nonzero fraction {preflight["control"]["positive_return_fraction"]:.2%}. '
      f'Old MMD mean: {preflight["old_mmd"]["model_return_mean"]:.5f}. '
      'The predefined gate (>=5% nonzero and std>0.01) passed before any parameter search. Zero return was not treated as a worst-case discovery.','',
      f'A separate 32-episode real-environment check with the same frozen actor returned mean '
      f'{preflight["real_actor_reference"]["real_environment_return_mean"]:.5f}, nonzero fraction '
      f'{preflight["real_actor_reference"]["positive_return_fraction"]:.2%}. '
      'All 1,600 reported environment rewards exactly matched visible reconstruction, without reading audit fields. '
      'This small real sample is supplementary and has different randomness; no ETT update changes the real actor here. '
      'The large model/environment gap remains unresolved. At zero residual the modeled next-state law follows its diagonal '
      'anchor conditioned on x_prime and has no dependence on x; this is a material off-diagonal baseline limitation.','',
      '## Exactly two losses, restricted optimization','',
      '`L_ETT = L_diag + lambda * mean(sum_{t=0}^{49} 0.95^t r_g(s_{t+1}))`, minimized. '
      'The sign is positive: reducing predicted task return is the pessimistic direction. '
      'The diagonal likelihood is the existing zero-displacement-atom plus continuous XY-density NLL, in nats per raw maze-unit 2D transition. '
      'There is no third loss, regularizer, weight decay, critic substitution, smooth reward or Lipschitz loss.','',
      f'**The diagonal backbone is frozen, so L_diag is constant:** training NLL {search["diagonal_train_nll_constant"]:.8f}; '
      f'held-out NLL {search["diagonal_validation_nll_constant"]:.8f}. '
      'This experiment verifies the second term in a restricted family; it is not joint two-loss training. '
      'Only six existing residual-head coordinates move: both output biases and both output weights at hidden-unit rows 0 and 32. '
      'The residual torso, all other head weights, nominal, actor and diagonal parameters remain fixed.','',
      'Two optimizer seeds each have lambda=0 and lambda=1 arms. Every arm executes 12 iterations, four independent N(0,I6) directions '
      'per iteration, sigma=0.2, and 16 complete paths at each plus/minus point. '
      'The same full-trajectory random keys are used within each plus/minus pair and between matched lambda arms. '
      'The estimate is `lambda * mean((J_plus-J_minus)*epsilon/(2*sigma))`; SGD subtracts 0.1 times it, '
      'with parameter-update L2 capped at 0.2. This step cap is an optimizer setting, not a loss. '
      'Each arm uses 1,536 parameter-query trajectories plus 208 fixed-monitor trajectories. '
      'Final iteration is evaluated; monitoring never selects a checkpoint or changes the fixed budget.','',
      'The estimator includes complete resampling of mixture choices, atoms, state-dependent action distributions, hard reward and geometry decisions. '
      'It estimates the derivative of a Gaussian-smoothed expected return in the six-dimensional parameter space, not the exact unsmoothed objective gradient. '
      'No ordinary pathwise gradient through a hard indicator or discrete draw is assumed. All estimated gradients were finite. '
      'Lambda=0 controls receive zero search gradient and remain exactly initialized, while spending the same sampling budget.','',
      'The existing L=0.25 anchored action-change bound, exact x=x_prime anchor branch, tanh residual, one-unit per-axis cap, '
      'endpoint projection and bound fallback are retained. They limit final displacement relative to the sampled diagonal anchor and ensure exact old-frame shift. '
      'They are not full physics validity, state-dependent absorbing-state guarantees, or all-pairs action Lipschitzness.','',
      '## Independent model evaluation','',
      'Four unused evaluation seeds (710001..710004), 128 complete paths each, give 512 paths per arm. '
      'They are independent of optimization directions, parameter-query draws, monitoring, and preflight. '
      'Control/treatment paths are paired by their random keys. All contexts are the same deterministic reset, '
      'so the bootstrap unit is an independent simulated path, not a source episode.','']
  for name in NAMES:
    m=evaluation['models'][name]
    lines.append(f'- {name}: mean return {m["model_return_mean"]:.5f}; nonzero {m["positive_return_fraction"]:.2%}; '
        f'rewarded steps {m["mean_rewarded_steps"]:.2f}; held-out diagonal NLL {m["heldout_diagonal_nll"]:.8f}.')
  for seed,p in evaluation['paired_lambda1_minus_lambda0'].items():
    lines.append(f'- {seed}, lambda1 minus lambda0: {p["mean"]:.5f}, paired model-path 95% interval '
        f'[{p["ci95_paired_model_paths"][0]:.5f}, {p["ci95_paired_model_paths"][1]:.5f}].')
  lines += ['', 'Both search seeds lower return on every evaluation-seed group. These intervals cover model Monte Carlo variation only; '
      'they do not validate physical outcomes, hidden-variable identification or worst-case Q. '
      'Both lambda=0 controls have identical final parameters and exactly identical evaluation paths under the common keys.','',
      '![Returns and reward timing](returns_and_reward_timing.png)','',
      '## Behavioral degradation checks','']
  for name in NAMES:
    m=evaluation['models'][name];p=paths[name]
    lines.append(f'- {name}: exact stationary steps {m["stationary_fraction"]:.2%}; mean motion {m["mean_motion"]:.4f}; '
        f'first rewarded step among paths that reach {p["first_reward_step_if_reached"]:.2f}; '
        f'near-grid-boundary steps {m["near_wall_boundary_fraction"]:.2%}; candidate geometry corrections {m["candidate_corrected_fraction"]:.2%}; '
        f'last ten steps exactly frozen on {p["exactly_stationary_last_10_steps_path_fraction"]:.2%} of paths.')
  lines += ['', 'Seed 0 mainly delays reward while increasing motion and leftward drift; its nonzero-return fraction actually increases. '
      'Seed 1 reduces the fraction reaching reward and increases boundary-related stagnation and projection use. '
      'Its 11.41% near-grid-boundary rate is a coordinate diagnostic, not an exact wall-contact label. '
      'Trajectory diversity does not collapse uniformly: the within-context trajectory XY spread rises from 1.052 to 1.436/1.611, '
      'but this includes divergence between successful and stalled paths and is not a quality score.','',
      'All final rollout samples pass endpoint, coordinate-cap, action-bound, F4-shift and original anchor-bound checks. '
      'Sampled straight segments still intersect blocked geometry on about 0.0234% / 0.0234% / 0.0039% of active reset transitions '
      '(control / seed0 / seed1). Axiswise simulator substeps and these approximate projection/segment checks are different; '
      'passing them is not proof of full reachability.','',
      '![Motion and geometry](motion_and_geometry.png)','',
      '![Mean trajectory positions](mean_trajectory_positions.png)','',
      'The continuation audit reuses 32 distinct-episode observable frozen non-reset starts and 32 moving starts from the previous development contexts. '
      'It uses four paths per start, key 720001, and only the remaining original episode steps. '
      'No death labels are loaded or inferred from stillness. On frozen-history starts, mean first-step motion rises '
      'from 0.0555 to 0.2052/0.2459 maze units. This is renewed arbitrary-motion evidence rather than preserved stationary behavior. '
      'The original control itself sometimes escapes those histories and earns positive model return; a frozen diagonal backbone does not solve multi-step model bias.','',
      '![Paired representative trajectories](paired_trajectory_examples.png)','',
      'Examples are the first three paired paths plus explicit return-extreme paths; indices and selection reasons are saved. '
      'All reset and continuation trajectories are retained for review, alongside separate observational-action/anchor diagnostics.','',
      '## Continue / stop and limits','',
      '**Stop before contrastive-agent alternating training.** This result establishes that finite-difference residual search can lower the specified '
      'hard model-rollout return without altering diagonal fit. It does not show that the chosen transition is a valid pessimistic outcome. '
      'The model/environment return gap, arbitrary motion on frozen histories, and boundary-related stagnation would contaminate actor learning if alternated now. '
      'Any follow-up should first test multi-step predictive validity and these trajectory pathologies on independently specified environment comparisons; '
      'it should not rerun the rejected raw-critic diagnosis. No joint diagonal/residual optimization or actor-critic alternation was launched.','',
      'The six-dimensional family and fixed 12-iteration budget do not exhaust the residual network or guarantee a global minimum. '
      'Independent rollout seeds only test model-internal optimization. There is no claim of identifying the true ETT or recovering a strict worst-case value.','',
      '## Reproduction and completed checks','',
      f'Completed: five numerical/contract tests; frozen-model signal preflight and 32 real-environment reference rollouts; '
      f'four matched-budget restricted-search arms; independent reset and continuation evaluation; saved-checkpoint diagonal NLL and sampling-identity checks. '
      f'Search and evaluation took {config["elapsed_seconds"]:.1f} seconds on {config["runtime"]["devices"]}. '
      'Freeze checks include parameter equality for diagonal and nominal, untouched residual torso/unselected head weights, '
      'and exact repeated fixed-key actor/nominal outputs. `config.json` hashes all preserved input artifacts. '
      'The separate artifact verification reloaded all four checkpoints and exactly reproduced 128 saved full paths from each, '
      'including states, actions and rewards. All checkpoints are Git-ignored. Its results are in `verification.json`.','',
      '```bash','python -m scripts.test_rollout_return',
      'python -m ett.run_rollout_return --phase preflight --out-dir artifacts/ett_rollout_return/preflight',
      config['command'],f'python -m ett.report_rollout_return --run-dir {out.as_posix()} --preflight-dir artifacts/ett_rollout_return/preflight',
      f'python -m scripts.check_rollout_return_artifacts --run-dir {out.as_posix()}','```','',
      'Use fresh output directories to reproduce. Checkpoints `s*_lambda*.pkl` stay local and are ignored by Git. '
      'Load one with `ett.anchored_transition.load_anchored`; it retains the existing sampling interface. '
      'The preflight driver hash predates removal of an unused local variable; all rollout computations are unchanged. '
      'The search run records the final driver hashes and `search_config.json` records the exact search-module hash. '
      'Code, configurations and evaluation artifacts are shareable; model checkpoints remain local.','',
      'The existing transition sampling interface is preserved. For already-loaded frozen policies and a batch of fixed-goal F4 states:',
      '','```python','import jax','from ett.anchored_transition import load_anchored',
      'model = load_anchored("artifacts/ett_rollout_return/residual6_s01/s0_lambda1.pkl")',
      'nominal_key, actor_key, transition_key = jax.random.split(jax.random.PRNGKey(901), 3)',
      'x_prime = nominal.sample(state, nominal_key, 1, goal=goal)  # [B, 2]',
      'x = actor(state, goal, actor_key)  # [B, 2], the execution action',
      'next_states = model.sample(state, x, x_prime, transition_key, num_samples=8, goal=goal)  # [B, 8, 8]',
      '```','',
      'Those eight next-state draws condition on the same x and x_prime. `TaskRollout.run` instead draws fresh independent '
      'policy actions for each trajectory and step, and returns arrays grouped as `[context, replicate, time, feature]`.']
  (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
  p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',required=True);p.add_argument('--preflight-dir',required=True);args=p.parse_args()
  out=Path(args.run_dir);config=read(out/'config.json')
  if config['status']!='complete':raise ValueError('search has not completed')
  evaluation=read(out/'reset_evaluation.json');history=read(out/'optimization_history.json')
  records=load_records(out);paths=describe_paths(records);write_json(out/'trajectory_diagnostics.json',paths)
  precision=reward_precision_audit(out)
  figures(out,records,evaluation,history)
  report(out,read(Path(args.preflight_dir)/'preflight.json'),config,read(out/'search_config.json'),evaluation,read(out/'continuation_evaluation.json'),paths,precision)
  write_json(out/'report_provenance.json',{'source_sha256':file_sha(__file__),'uses_saved_model_trajectories_only':True,
      'additional_deterministic_environment_boundary_steps':3,
      'command':f'python -m ett.report_rollout_return --run-dir {out.as_posix()} --preflight-dir {args.preflight_dir}'})
  print('wrote rollout-return report and four figures',flush=True)


if __name__=='__main__':main()
