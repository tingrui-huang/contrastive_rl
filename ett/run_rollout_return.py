"""Bounded hard-return experiment with frozen policies and diagonal backbone."""
import argparse
import json
from pathlib import Path
import pickle
import re
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from crl.envs import TwoRouteSwampWindyF4Env
from ett.anchored_transition import load_anchored,save_anchored
from ett.dataset import ExpertTransitionDataset
from ett.eval_diagonal_transition import _log_prob,_open_endpoint
from ett.rollout_return import (START,GOAL,HORIZON,DISCOUNT,SEARCH_DIMENSION,
    TaskRollout,task_reward,antithetic_gradient,descent_step,residual_parameters,two_loss_objective)
from ett.run_distribution_matching import frozen_actor,write_json
from ett.train_diagonal_transition import _git_provenance
from propensity.nominal_policy import load_nominal_policy
from scripts.make_swamp_f4_failure_bank import file_sha

SOURCE=Path('artifacts/ett_distribution_matching/f4_p30_s01_guarded')
PREVIOUS_OBJECTIVE=Path('artifacts/ett_objective_diagnostic/f4_p30_s01')


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def tree_equal(left,right):
  return jax.tree.structure(left)==jax.tree.structure(right) and all(
      np.array_equal(a,b) for a,b in zip(jax.tree.leaves(left),jax.tree.leaves(right)))


def setup(out):
  prior=read(SOURCE/'config.json');args=prior['arguments']
  paths={'dataset':Path(args['dataset']),'nominal':Path(args['nominal'])/'best.pkl',
      'actor':Path(args['actor']), 'control':SOURCE/'s0_L0p25_lambda0/final.pkl',
      'old_mmd':SOURCE/'s0_L0p25_lambda1/final.pkl'}
  paths['actor_training_log']=Path('artifacts/f4_p30_server_30076/results/logs/f4_p30_sweep/p30_a0_a01_a03_s0_s1/alpha0_seed0_train.log')
  missing={name:str(p) for name,p in paths.items() if not p.is_file()}
  if missing:
    write_json(out/'blocked.json',{'missing_dependencies':missing,'search_started':False})
    raise FileNotFoundError(missing)
  if file_sha(paths['dataset'])!=prior['dataset']['dataset']['sha256']:
    raise ValueError('dataset differs from authoritative run')
  for path,expected in prior['input_checkpoint_sha256'].items():
    if file_sha(path)!=expected:raise ValueError(f'frozen checkpoint changed: {path}')
  old_metrics=read(SOURCE/'metrics.json')
  for name,arm in [('control','s0_L0p25_lambda0'),('old_mmd','s0_L0p25_lambda1')]:
    if file_sha(paths[name])!=old_metrics['arms'][arm]['checkpoint_sha256']:
      raise ValueError('guarded checkpoint changed')
  log=paths['actor_training_log'].read_text(encoding='utf-8')
  discounts=re.findall(r'\bdiscount=([0-9.]+)',log)
  if not discounts or any(float(v)!=DISCOUNT for v in discounts):
    raise ValueError('actor training discount does not match rollout discount')
  dataset=ExpertTransitionDataset(str(paths['dataset']))
  if dataset.report['split']!=prior['dataset']['split']:raise ValueError('episode split changed')
  if dataset.metadata['max_episode_steps']!=50:raise ValueError('episode horizon differs')
  if not np.all(dataset.arrays('all').goal==GOAL):raise ValueError('commanded goal differs')
  resets=dataset.arrays('all').state[dataset.arrays('all').timestep==0]
  if not np.all(resets==START):raise ValueError('fixed task reset changed')
  nominal=load_nominal_policy(args['nominal'])
  if nominal.spec.conditioning!='state_goal':raise ValueError('nominal goal convention changed')
  actor,actor_info=frozen_actor(args['actor'],prior['references']['dataset_content_sha256'])
  actor_info['sampling']='fresh independent agent action per trajectory per step; frozen state/commanded-goal policy'
  model=load_anchored(paths['control'])
  if model.action_bound!=.25 or any(np.any(np.asarray(v)!=0) for v in model.params['residual']['linear'].values()):
    raise ValueError('expected preserved zero-residual L=0.25 control')
  hashes={str(p).replace('\\','/'):file_sha(p) for p in paths.values()}
  for root in (SOURCE,PREVIOUS_OBJECTIVE,Path('artifacts/critic_continuation/f4_p30_alpha_s01'),Path('artifacts/failure_scoring/scope_b_f4_p30_s01')):
    hashes.update({p.as_posix():file_sha(p) for p in root.rglob('*') if p.is_file()})
  return model,nominal,actor,dataset,{'paths':paths,'input_file_sha256':hashes,
      'actor':actor_info,'dataset_split':dataset.report['split'],
      'nominal_conditioning':nominal.spec.conditioning,'discount':DISCOUNT,
      'discount_source':paths['actor_training_log'], 'horizon':50,'commanded_goal':GOAL,
      'reward':'1[newest XY distance to fixed goal < 2.0]; repeat reward, no goal termination',
      'reward_justification':'fixed reward region and all absorbing swamp cells are disjoint; not a general reward reconstruction',
      'critic_diagnostics_read_not_repeated':['e670a33b289e55c9b1eaff1e4a92e6e1f443eee6','6947186b3b85c475ccd3cb06280d182cf7ae58f2','089f0839edab5382e666037589146bc2d76379ac'],
      'hidden_labels_or_failure_bank_used':False,'critic_or_scorer_used':False}


def rollout_metrics(record):
  states=record['states'];before,after=states[:,:,:-1],states[:,:,1:]
  active=record['active'];delta=after[...,:2]-before[...,:2]
  active_count=int(active.sum())
  motion=np.linalg.norm(delta,axis=-1)
  segment=before[...,:2][...,None,:]+np.linspace(0,1,21)[None,None,None,:,None]*delta[...,None,:]
  frozen=np.all(before.reshape(before.shape[:-1]+(4,2))==before[...,:2][...,None,:],axis=(-1,-2))
  ret=record['return'].ravel()
  result={'paths':len(ret),'active_transitions':active_count,'model_return_mean':float(ret.mean()),
      'model_return_std':float(ret.std()),'model_return_min':float(ret.min()),'model_return_max':float(ret.max()),
      'positive_return_fraction':float((ret>0).mean()),
      'reached_0p5_fraction':float((np.linalg.norm(after[...,:2]-GOAL[:2],axis=-1)<.5).any(-1).mean()),
      'mean_rewarded_steps':float(record['reward'].sum(-1).mean()),
      'mean_motion':float(motion[active].mean()),'stationary_fraction':float((motion[active]==0).mean()),
      'near_stationary_fraction':float((motion[active]<=.05).mean()),
      'mean_final_x':float(states[:,:,-1,0].mean()),
      'final_xy_std':states[:,:,-1,:2].reshape(-1,2).std(0),
      'mean_within_context_trajectory_xy_std':float(np.linalg.norm(after[...,:2].std(1),axis=-1).mean()),
      'safe_route_fraction':float((after[...,1]<2.).any(-1).mean()),
      'leftward_step_fraction':float((delta[...,0][active]<-.01).mean()),
      'near_wall_boundary_fraction':float((np.minimum(np.mod(after[...,:2],1),1-np.mod(after[...,:2],1)).min(-1)[active]<.02).mean()),
      'history_shift_max_error':float(np.abs(after[...,2:]-before[...,:6])[active].max(initial=0)),
      'endpoint_invalid_count':int(((~_open_endpoint(after[...,:2]))&active).sum()),
      'segment_blocked_fraction':float((~_open_endpoint(segment)).any(-1)[active].mean()),
      'coordinate_cap_violation_count':int(((np.abs(delta)>1.+1e-6).any(-1)&active).sum()),
      'agent_action_violation_count':int(((np.abs(record['action'])>1.).any(-1)&active).sum()),
      'aux_action_violation_count':int(((np.abs(record['aux_observational_action'])>1.).any(-1)&active).sum()),
      'anchor_bound_violation_count':int(((record['emitted_change']>record['radius']+1e-7)&active).sum()),
      'bound_fallback_fraction':float(record['bound_fallback'][active].mean()),
      'candidate_corrected_fraction':float(record['candidate_corrected'][active].mean()),
      'frozen_stack_active_rows':int((frozen&active).sum()),
      'moves_from_frozen_stack_fraction':float((motion[frozen&active]>.05).mean()) if (frozen&active).any() else None}
  for name in ('history_shift_max_error','endpoint_invalid_count','coordinate_cap_violation_count',
               'agent_action_violation_count','aux_action_violation_count','anchor_bound_violation_count'):
    if result[name]!=0:raise RuntimeError(f'rollout contract violation: {name}={result[name]}')
  if not np.all(record['commanded_goal']==GOAL):raise RuntimeError('goal changed')
  if not all(np.isfinite(v).all() for v in record.values()):raise RuntimeError('non-finite rollout')
  return result


def save_rollouts(out,name,record):
  main={k:record[k] for k in ('states','action','reward','active','return','commanded_goal')}
  auxiliary={k:v for k,v in record.items() if k not in main}
  np.savez_compressed(out/f'{name}_trajectories.npz',**main)
  np.savez_compressed(out/f'{name}_auxiliary.npz',**auxiliary)


def concatenate_rollouts(records):
  # Same start contexts, independent trajectory replicates from each key.
  return {k:np.concatenate([r[k] for r in records],axis=1) for k in records[0]}


def real_actor_check(actor,count=32):
  trajectories=[];actions=[];rewards=[]
  for episode in range(count):
    env=TwoRouteSwampWindyF4Env(seed=820000+episode)
    obs=env.reset();states=[obs[:8]];aa=[];rr=[]
    for t in range(50):
      action=np.asarray(actor(obs[None,:8],obs[None,8:],jax.random.PRNGKey(830000+episode*50+t)))[0]
      obs,reward,done,_=env.step(action)
      if reward!=float(task_reward(obs[:8],obs[8:])) or done or not np.all(obs[8:]==GOAL):
        raise RuntimeError('fixed-task reward/termination reconstruction failed')
      states.append(obs[:8]);aa.append(action);rr.append(reward)
    trajectories.append(states);actions.append(aa);rewards.append(rr)
  returns=np.sum(np.asarray(rewards)*DISCOUNT**np.arange(50),-1)
  return {'states':np.array(trajectories),'action':np.array(actions),'reward':np.array(rewards),'return':returns}, {
      'episodes':count,'real_environment_return_mean':float(returns.mean()),'real_environment_return_std':float(returns.std()),
      'positive_return_fraction':float((returns>0).mean()),'reward_matches_visible_reconstruction':True,
      'scope':'supplementary fixed-actor real-environment runs; no ETT involvement or actor improvement claim; hidden fields not read'}


def preflight(out,model,nominal,actor):
  results={};zero=np.zeros(6,np.float32)
  for name,current in [('control',model),('old_mmd',load_anchored(SOURCE/'s0_L0p25_lambda1/final.pkl'))]:
    engine=TaskRollout(current,nominal,actor)
    records=[engine.run(zero,START[None],GOAL[None],jax.random.PRNGKey(seed),128) for seed in (410001,410002)]
    record=concatenate_rollouts(records)
    results[name]=rollout_metrics(record)
    save_rollouts(out,name,record)
    print(f'preflight {name}: mean return={results[name]["model_return_mean"]:.5f}, nonzero={results[name]["positive_return_fraction"]:.3f}',flush=True)
  real,results['real_actor_reference']=real_actor_check(actor)
  np.savez_compressed(out/'real_actor_trajectories.npz',**real)
  baseline=results['control']
  results['search_signal_gate_passed']=bool(baseline['positive_return_fraction']>=.05 and baseline['model_return_std']>.01)
  results['gate_rule']='control: at least 5% nonzero full-episode returns and std >0.01; failure means stop and diagnose, not a worst-case success'
  write_json(out/'preflight.json',results)
  return results


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--phase',choices=('preflight','search'),required=True)
  parser.add_argument('--out-dir',required=True)
  parser.add_argument('--preflight-dir')
  parser.add_argument('--iterations',type=int,default=12)
  parser.add_argument('--directions',type=int,default=4)
  parser.add_argument('--search-replicates',type=int,default=16)
  parser.add_argument('--seeds',default='0,1')
  args=parser.parse_args();out=Path(args.out_dir)
  if out.exists():raise ValueError('use a fresh output directory')
  out.mkdir(parents=True);started=time.time()
  model,nominal,actor,dataset,config=setup(out)
  config.update({'arguments':vars(args),'command':'python -m ett.run_rollout_return '+' '.join(sys.argv[1:]),
      'git':_git_provenance(),'runtime':{'jax':jax.__version__,'devices':list(map(str,jax.devices()))},
      'status':'running','source_file_sha256':{p:file_sha(p) for p in ('ett/rollout_return.py','ett/run_rollout_return.py','scripts/test_rollout_return.py')}})
  write_json(out/'config.json',config)
  if args.phase=='preflight':preflight(out,model,nominal,actor)
  else:
    from ett.search_rollout_return import search
    search(out,args,model,nominal,actor,dataset)
  config['elapsed_seconds']=time.time()-started
  config['all_previous_inputs_unchanged']=all(file_sha(p)==sha for p,sha in config['input_file_sha256'].items())
  if not config['all_previous_inputs_unchanged']:raise RuntimeError('a previous artifact changed')
  config['status']='complete';write_json(out/'config.json',config)
  print(f'completed {args.phase} in {config["elapsed_seconds"]:.1f}s',flush=True)


if __name__=='__main__':main()
