"""Bounded ETT action-response evaluation, including paired simulator references."""
import argparse
import json
from pathlib import Path
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from ett.action_response import (ACTION_NAMES,GROUPS,SELECTION_RULES,InstrumentedProbe,
                                actions_for,select_contexts,response_metrics)
from ett.action_reference import teacher_contexts,prescribed_reference,continuation_indices
from ett.anchored_transition import load_anchored
from ett.dataset import ExpertTransitionDataset
from ett.rollout_return import GOAL,task_reward
from ett.run_rollout_return import read
from ett.run_distribution_matching import write_json
from propensity.nominal_policy import load_nominal_policy
from scripts.make_swamp_f4_failure_bank import file_sha


SOURCE=Path('artifacts/ett_rollout_return/residual6_s01')
MODEL_FILES={'control':'s0_lambda0.pkl','return_seed0':'s0_lambda1.pkl','return_seed1':'s1_lambda1.pkl'}
PRIOR_REPORTS=[SOURCE/'REPORT.md',Path('artifacts/ett_distribution_matching/f4_p30_s01_guarded/REPORT.md'),
    Path('artifacts/ett_objective_diagnostic/f4_p30_s01/REPORT.md'),
    Path('artifacts/critic_continuation/f4_p30_alpha_s01/REPORT.md')]


def short_model_rollout(model,nominal,state,goal,initial_xp,actions,count,horizon=5):
  """Prescribed x repeated for five steps; fresh nominal x_prime after step 0.

  Only the first observational action is paired to the restored teacher state.
  Later nominal draws are model auxiliary variables, not oracle observations.
  """
  @jax.jit
  def run(state,goal,xp0,execution,key):
    def step(s,item):
      t,key=item;nk,tk=jax.random.split(key)
      xp=jnp.where(t==0,xp0,nominal.sample(s,nk,1,goal=goal))
      output,detail=model.sample_flat(model.params,s,execution,xp,goal,tk,1)
      next_state=output[:,0]
      return next_state,{'state':next_state,'observational_action':xp,'reward':task_reward(next_state,goal),
          'candidate_corrected':detail['candidate_corrected'][:,0],'bound_fallback':detail['bound_fallback'][:,0]}
    _,records=jax.lax.scan(step,state,(jnp.arange(horizon),jax.random.split(key,horizon)))
    return {k:jnp.swapaxes(v,0,1) for k,v in records.items()}
  repeat=lambda a:np.repeat(a,count,axis=0)
  outputs=[]
  for a in range(actions.shape[1]):
    values=run(repeat(state),repeat(goal),repeat(initial_xp),repeat(actions[:,a]),jax.random.PRNGKey(960001))
    values={k:np.asarray(v).reshape((len(state),count)+v.shape[1:]) for k,v in values.items()}
    values['states']=np.concatenate([np.broadcast_to(state[:,None,None],(len(state),count,1,8)),values.pop('state')],axis=2)
    if not np.array_equal(values['states'][...,1:,2:],values['states'][...,:-1,:6]):raise RuntimeError('continuation F4 shift failed')
    outputs.append(values)
  return {k:np.stack([v[k] for v in outputs],axis=1) for k in outputs[0]}


def grouped_metrics(record,groups):
  return {'all':response_metrics(record),**{name:response_metrics({k:v[groups==name] for k,v in record.items()})
      for name in GROUPS if (groups==name).any()}}


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--out-dir',required=True)
  parser.add_argument('--contexts-per-group',type=int,default=8)
  parser.add_argument('--samples',type=int,default=64)
  parser.add_argument('--sim-episodes',type=int,default=64)
  parser.add_argument('--sim-contexts-per-group',type=int,default=4)
  args=parser.parse_args();out=Path(args.out_dir)
  if out.exists():raise ValueError('use a fresh diagnostic directory')
  if not (1<=args.contexts_per_group<=8 and 2<=args.samples<=64 and 1<=args.sim_episodes<=64 and 1<=args.sim_contexts_per_group<=4):
    raise ValueError('diagnostic budget exceeds allowed bounds or is empty')
  out.mkdir(parents=True);start=time.time()
  prior=read(SOURCE/'config.json');evaluation=read(SOURCE/'reset_evaluation.json')
  paths={'dataset':Path(prior['paths']['dataset']),'nominal':Path(prior['paths']['nominal']),
         **{name:SOURCE/path for name,path in MODEL_FILES.items()}}
  missing={k:str(v) for k,v in paths.items() if not v.is_file()}
  if missing:
    write_json(out/'blocked.json',{'missing_dependencies':missing,'training_started':False})
    raise FileNotFoundError(missing)
  hashes={str(p).replace('\\','/'):file_sha(p) for p in paths.values()}
  for root in (SOURCE.parent,Path('artifacts/ett_distribution_matching/f4_p30_s01_guarded'),
               Path('artifacts/ett_objective_diagnostic/f4_p30_s01')):
    hashes.update({p.as_posix():file_sha(p) for p in root.rglob('*') if p.is_file()})
  for p in PRIOR_REPORTS:hashes[p.as_posix()]=file_sha(p)
  for path,expected in prior['input_file_sha256'].items():
    if file_sha(path)!=expected:raise RuntimeError(f'prior dependency changed: {path}')
  for name,filename in MODEL_FILES.items():
    if file_sha(paths[name])!=evaluation['models'][Path(filename).stem]['checkpoint_sha256']:
      raise RuntimeError(f'checkpoint hash mismatch: {name}')
  config={'status':'running','reviewed_commit':'03f7686a2277c684d3801a6dcb7f24bd81ea622c',
      'branch':'feature/pointmaze-causal-transition','remote_head_checked':'03f7686a2277c684d3801a6dcb7f24bd81ea622c',
      'command':'python -m ett.run_action_response '+' '.join(sys.argv[1:]),'arguments':vars(args),
      'input_sha256':hashes,'checkpoint_paths':paths,'selection_rules':SELECTION_RULES,
      'selection_seed':940000,'nominal_draw_seed':950000,'model_probe_key':950001,
      'sim_probe_key':950002,'continuation_key':960001,'continuation_exogenous_base_seed':970000,
      'nominal_draws_per_data_context':2,'transition_draws_per_probe':args.samples,'actions':ACTION_NAMES,
      'one_step_budget_max':3*10*args.samples*(4*args.contexts_per_group*2+4*args.sim_contexts_per_group),
      'continuation_horizon':5,'prescribed_continuation_actions':[[0,0],[.5,0],[-.5,0]],
      'objective_unchanged':'L_diag + lambda * L_off; exactly two terms; this task computes no optimization gradients or updates',
      'critic_diagnostic_repeated':False,'training_started':False,'runtime':{'jax':jax.__version__,'devices':list(map(str,jax.devices()))}}
  write_json(out/'config.json',config)
  dataset=ExpertTransitionDataset(str(paths['dataset']));arrays=dataset.arrays('validation')
  nominal=load_nominal_policy(paths['nominal'].parent)
  if nominal.spec.conditioning!='state_goal':raise RuntimeError('nominal goal convention changed')
  models={name:load_anchored(path) for name,path in paths.items() if name in MODEL_FILES}
  if any(np.any(np.asarray(v)!=0) for v in models['control'].params['residual']['linear'].values()):raise RuntimeError('control is not zero residual')
  ids,groups,eligible=select_contexts(arrays.state,arrays.episode_id,arrays.timestep,args.contexts_per_group,940000)
  contexts=arrays.take(ids)
  xp=np.asarray(nominal.sample(contexts.state,jax.random.PRNGKey(950000),2,goal=contexts.goal)).reshape(-1,2)
  state=np.repeat(contexts.state,2,0);goal=np.repeat(contexts.goal,2,0);probe_groups=np.repeat(groups,2)
  write_json(out/'contexts.json',{'rules':SELECTION_RULES,'eligible':eligible,'selection_seed':940000,'groups':groups,
      'validation_row_indices':ids,'episode_id':contexts.episode_id,'timestep':contexts.timestep,'state':contexts.state,
      'commanded_goal':contexts.goal,'nominal_observational_actions':xp.reshape(-1,2,2),
      'teacher_population':dataset.population_manifest,'source_metadata':{k:dataset.metadata[k] for k in ('random_frac','force_safe_prob','teacher_noise')},
      'conditioning':'visible F4 + unchanged commanded goal; stationary history is not a death label'})
  results={};checks={}
  for name,model in models.items():
    record,check=InstrumentedProbe(model).run(state,xp,goal,jax.random.PRNGKey(950001),args.samples)
    np.savez_compressed(out/f'{name}_model_probes.npz',**record)
    results[name]=grouped_metrics(record,probe_groups);checks[name]=check
    if name=='control' and results[name]['all']['max_fixed_key_execution_action_effect']>2e-6:raise RuntimeError('control action invariance failed')
    print(name,'model action effect',results[name]['all']['max_fixed_key_execution_action_effect'],flush=True)
  write_json(out/'model_probe_metrics.json',results)
  rows,sim_groups,protocol=teacher_contexts(args.sim_episodes,args.sim_contexts_per_group)
  write_json(out/'simulator_protocol.json',protocol)
  if not rows:raise RuntimeError('no simulator contexts selected')
  sim_state=np.array([r['state'] for r in rows]);sim_goal=np.array([r['goal'] for r in rows]);natural_xp=np.array([r['observational_action'] for r in rows])
  sim_actions=actions_for(natural_xp)
  write_json(out/'simulator_restoration_audit.json',{'scope':'evaluation-only hidden restoration; never loaded by model probes',
      'groups':sim_groups,'contexts':rows})
  reference,reference_audit=prescribed_reference(rows,sim_actions,horizon=1)
  np.savez_compressed(out/'simulator_one_step.npz',**reference,state=sim_state,goal=sim_goal,observational_action=natural_xp)
  for i,row in enumerate(rows):
    if not np.array_equal(reference['states'][i,-1,1],row['observed_next_state']):raise RuntimeError('natural action branch replay failed')
  comparison={}
  for name,model in models.items():
    record,check=InstrumentedProbe(model).run(sim_state,natural_xp,sim_goal,jax.random.PRNGKey(950002),args.samples)
    np.savez_compressed(out/f'{name}_teacher_probes.npz',**record);checks[name+'_teacher']=check
    predicted=record['next_state'][...,:2].mean(2);actual=reference['states'][:,:,1,:2]
    response_error=(predicted-predicted[:,-1:])-(actual-actual[:,-1:])
    comparison[name]={'probe_metrics':grouped_metrics(record,sim_groups),
        'mean_position_discrepancy_to_single_paired_outcome':float(np.linalg.norm(predicted-actual,axis=-1).mean()),
        'mean_action_contrast_discrepancy_to_single_paired_outcome':float(np.linalg.norm(response_error,axis=-1).mean()),
        'per_group_contrast_discrepancy':{g:float(np.linalg.norm(response_error[sim_groups==g],axis=-1).mean()) for g in GROUPS if (sim_groups==g).any()},
        'scope':'descriptive disagreement with a single restored underlying context, NOT conditional ETT estimation or a required equality'}
  write_json(out/'simulator_comparison.json',comparison)
  # Prescribed action sequences, not actor closed-loop rollouts and not data continuations.
  eligible=continuation_indices(rows,5);continuation_rows=[rows[i] for i in eligible]
  selected_actions=sim_actions[eligible,:3]
  write_json(out/'continuation_contexts.json',{'selection_rule':'selected one-step contexts with timestep + 5 <= 50; no outcome filtering',
      'one_step_context_indices':eligible,'groups':sim_groups[eligible],
      'episode':[r['episode'] for r in continuation_rows],'timestep':[r['timestep'] for r in continuation_rows],
      'excluded_one_step_context_indices':np.setdiff1d(np.arange(len(rows)),eligible)})
  continuation,hidden_audit=prescribed_reference(continuation_rows,selected_actions,horizon=5)
  np.savez_compressed(out/'simulator_continuation.npz',**continuation)
  write_json(out/'continuation_hidden_audit.json',{'scope':'evaluation only; separate from model inputs','records':hidden_audit})
  continuation_metrics={}
  for name,model in models.items():
    record=short_model_rollout(model,nominal,sim_state[eligible],sim_goal[eligible],natural_xp[eligible],selected_actions,min(args.samples,32))
    np.savez_compressed(out/f'{name}_continuation.npz',**{k:v for k,v in record.items() if k!='observational_action'})
    np.savez_compressed(out/f'{name}_continuation_auxiliary.npz',observational_action=record['observational_action'])
    mean_xy=record['states'][...,:2].astype(np.float64).mean(2);ref_xy=continuation['states'][...,:2]
    discrepancy=np.linalg.norm(mean_xy-ref_xy,axis=-1)
    response=(mean_xy[:,1]-mean_xy[:,2])-(ref_xy[:,1]-ref_xy[:,2])
    continuation_metrics[name]={'mean_position_discrepancy_by_step':discrepancy.mean((0,1)),
        'opposing_action_contrast_discrepancy_by_step':np.linalg.norm(response,axis=-1).mean(0),
        'candidate_corrected_by_step':record['candidate_corrected'].mean((0,1,2)),
        'bound_fallback_by_step':record['bound_fallback'].mean((0,1,2)),
        'exact_f4_shift':True,'model_paths':len(eligible)*3*min(args.samples,32),
        'scope':'first x_prime is the natural teacher action at the paired start; later x_prime sampled from nominal at model states; single underlying simulator context and one future noise tape per start'}
  write_json(out/'continuation_metrics.json',continuation_metrics)
  write_json(out/'numerical_checks.json',checks)
  config.update(status='complete',elapsed_seconds=time.time()-start,
      all_previous_inputs_unchanged=all(file_sha(p)==v for p,v in hashes.items()),
      source_sha256={p:file_sha(p) for p in ('ett/action_response.py','ett/action_reference.py','ett/run_action_response.py')})
  if not config['all_previous_inputs_unchanged']:raise RuntimeError('previous artifact modified')
  write_json(out/'config.json',config)
  print(f'completed action-response diagnostic in {config["elapsed_seconds"]:.1f}s',flush=True)


if __name__=='__main__':main()
