"""Four-weight, exactly zero-bias ETT return-search ablation. No architecture change."""
import argparse
from pathlib import Path
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from ett.action_response import InstrumentedProbe,GROUPS,response_metrics
from ett.anchored_transition import load_anchored,save_anchored
from ett.eval_diagonal_transition import _log_prob
from ett.rollout_return import (TaskRollout,START,GOAL,HEAD_ROWS,residual_parameters,
    antithetic_gradient,descent_step,two_loss_objective)
from ett.run_rollout_return import (setup,read,tree_equal,rollout_metrics,save_rollouts,concatenate_rollouts)
from ett.search_rollout_return import return_interval,continuation_contexts
from ett.run_distribution_matching import write_json
from ett.train_diagonal_transition import _git_provenance
from scripts.make_swamp_f4_failure_bank import file_sha


PRIOR=Path('artifacts/ett_rollout_return/residual6_s01')
PROBES=Path('artifacts/ett_action_response/f4_return_s01_horizon50')
OLD={'control':'s0_lambda0','six_s0':'s0_lambda1','six_s1':'s1_lambda1'}
PROBE_NAMES={'control':'control','six_s0':'return_seed0','six_s1':'return_seed1'}


def embed_weights(weights):
  """Map four selected weight coordinates to the existing six-coordinate API."""
  if weights.shape!=(4,):raise ValueError('expected four output-weight coordinates')
  return jnp.concatenate([jnp.zeros(2,jnp.float32),jnp.asarray(weights,jnp.float32)])


def require_zero_initial(model):
  if model.action_bound!=.25:raise ValueError('action bound changed')
  if any(np.any(np.asarray(v)!=0) for v in model.params['residual']['linear'].values()):
    raise ValueError('must start from the original zero head, not a searched model with biases removed')


def check_subspace(initial,actual,weights):
  expected=residual_parameters(initial,embed_weights(weights))
  if not tree_equal(actual,expected):raise RuntimeError('parameters outside selected subspace changed')
  if not np.all(np.asarray(actual['residual']['linear']['b'])==0):raise RuntimeError('output bias is not exactly zero')
  if not tree_equal(actual['diagonal'],initial['diagonal']):raise RuntimeError('diagonal changed')
  return True


def load_arrays(path):
  with np.load(path,allow_pickle=False) as data:return {k:data[k] for k in data.files}


def require_saved_equal(actual,path):
  saved=load_arrays(path)
  for key,value in saved.items():
    if not np.array_equal(actual[key],value):raise RuntimeError(f'fixed-key reproduction failed: {path}/{key}')


def optimize(engine,initial,seed,diagonal_nll,smoke=False):
  iterations,directions,replicates=(1,1,4) if smoke else (12,4,16)
  # These are exactly the weight-coordinate components of the old iid N(0,I6)
  # draws. Their marginal law is iid N(0,I4); shared coordinates reduce noise.
  noise=np.random.default_rng(600000+seed).normal(size=(12,4,6)).astype(np.float32)[:iterations,:directions,2:]
  weights=np.zeros(4,np.float32)
  def score(w,key):
    record=engine.run(embed_weights(w),START[None],GOAL[None],jax.random.PRNGKey(key),replicates)
    rollout_metrics(record)
    return float(record['return'].mean())
  history=[{'iteration':0,'weights':weights.copy(),'center_monitor_return':score(weights,620000+seed)}]
  for iteration in range(iterations):
    plus=[];minus=[]
    for direction,epsilon in enumerate(noise[iteration]):
      key=610000+seed*10000+iteration*100+direction
      for sign,values in ((1,plus),(-1,minus)):
        trial=weights+sign*.2*epsilon
        params=residual_parameters(initial,embed_weights(trial));check_subspace(initial,params,trial)
        values.append(score(trial,key))
    gradient=antithetic_gradient(plus,minus,noise[iteration],.2,1.)
    if not np.isfinite(gradient).all():raise RuntimeError('nonfinite gradient')
    weights=descent_step(weights,gradient,.1,.2)
    check_subspace(initial,residual_parameters(initial,embed_weights(weights)),weights)
    monitor=score(weights,620000+seed)
    total,terms=two_loss_objective(diagonal_nll,jnp.array([monitor]),1.)
    history.append({'iteration':iteration+1,'weights':weights.copy(),'bias':[0.,0.],
        'positive_returns':plus,'negative_returns':minus,'gradient':gradient,
        'center_monitor_return':monitor,'diagonal_loss':float(terms[0]),'off_loss':float(terms[1]),'two_term_loss':float(total)})
    if (iteration+1)%3==0 or smoke:print(f'four_s{seed} iteration={iteration+1} return={monitor:.4f}',flush=True)
  return weights,history,noise


def main():
  parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out-dir',required=True)
  parser.add_argument('--smoke',action='store_true',help='one update for seed 0; not a comparison result')
  args=parser.parse_args();out=Path(args.out_dir)
  if out.exists():raise ValueError('use a fresh output directory')
  out.mkdir(parents=True);start=time.time()
  extra=[PRIOR/f'{name}.pkl' for name in OLD.values()]+[PRIOR/'search_config.json',PROBES/'contexts.json']
  missing=[p.as_posix() for p in extra if not p.is_file()]
  if missing:
    write_json(out/'blocked.json',{'missing_dependencies':missing,'training_succeeded':False});raise FileNotFoundError(missing)
  model,nominal,actor,dataset,config=setup(out);require_zero_initial(model);initial=model.params
  prior_search=read(PRIOR/'search_config.json');gate=read(PRIOR.parent/'preflight/preflight.json')
  if not gate['search_signal_gate_passed']:raise RuntimeError('original return signal gate did not pass')
  hashes=config['input_file_sha256']
  for root in (PRIOR.parent,PROBES):hashes.update({p.as_posix():file_sha(p) for p in root.rglob('*') if p.is_file()})
  config.update(status='running',git=_git_provenance(),command='python -m ett.bias_ablation '+' '.join(sys.argv[1:]),
      smoke=args.smoke,reviewed_commits=['03f7686a2277c684d3801a6dcb7f24bd81ea622c','b2e65ad41dc6ed60525dee646d85cace18d22a00'],
      objective='L_diag + lambda * mean(sum_t 0.95**t r(s_next)), lambda=1, minimized; exactly two losses',
      first_loss_constant=True,joint_two_loss_training=False,parameter_dimension=4,weight_rows=HEAD_ROWS,biases_fixed_at=[0,0],
      iterations=1 if args.smoke else 12,directions_per_iteration=1 if args.smoke else 4,
      trajectories_per_signed_query=4 if args.smoke else 16,sigma=.2,learning_rate=.1,update_norm_cap=.2,
      optimizer_seeds=[0] if args.smoke else [0,1],selection='final iteration only; monitor never selects or tunes',
      perturbations='iid N(0,I4), using columns 2:6 of original seed-matched N(0,I6) draws',
      search_geometry_caveat='Removing bias dimensions changes perturbation geometry and update-norm clipping; not isolation of every optimization effect',
      evaluation_keys=[710001,710002,710003,710004],evaluation_paths_per_key=128,
      runtime={'jax':jax.__version__,'devices':list(map(str,jax.devices()))})
  write_json(out/'config.json',config)
  nominal_copy=jax.tree.map(np.array,nominal.params)
  actor_before=np.asarray(actor(START[None],GOAL[None],jax.random.PRNGKey(990)))
  validation=dataset.arrays('validation');nll=float(-_log_prob(model.diagonal,validation,4096).mean())
  train_nll=float(-_log_prob(model.diagonal,dataset.arrays('train'),4096).mean())
  if nll!=prior_search['diagonal_validation_nll_constant'] or train_nll!=prior_search['diagonal_train_nll_constant']:
    raise RuntimeError('diagonal likelihood compatibility failed')
  config.update(diagonal_train_nll_constant=train_nll,diagonal_validation_nll_constant=nll)
  models={name:load_anchored(PRIOR/f'{old}.pkl') for name,old in OLD.items()}
  if not tree_equal(models['control'].params,initial):raise RuntimeError('old control differs from original initialization')
  engine=TaskRollout(model,nominal,actor);histories={};new_weights={};compatibility={}
  # Verify unchanged control and searched checkpoints before reusing their data.
  contexts,groups=continuation_contexts(dataset)
  old_contexts=read(PRIOR/'continuation_evaluation.json')
  if not np.array_equal(contexts.episode_id,old_contexts['episode_id']) or not np.array_equal(contexts.timestep,old_contexts['timestep']):
    raise RuntimeError('continuation context selection changed')
  for name,current in models.items():
    current_engine=TaskRollout(current,nominal,actor)
    reset=concatenate_rollouts([current_engine.run(np.zeros(6,np.float32),START[None],GOAL[None],jax.random.PRNGKey(k),128) for k in config['evaluation_keys']])
    require_saved_equal(reset,PRIOR/f'{OLD[name]}_trajectories.npz')
    continuation=current_engine.run(np.zeros(6,np.float32),contexts.state,contexts.goal,jax.random.PRNGKey(720001),4,50-contexts.timestep)
    require_saved_equal(continuation,PRIOR/f'{OLD[name]}_continuation_trajectories.npz')
    compatibility[name]={'reset_paths_reproduced_exactly':512,'remaining_horizon_continuation_paths_reproduced_exactly':256}
  print('old reset and remaining-horizon continuations reproduced exactly',flush=True)
  for seed in config['optimizer_seeds']:
    weights,history,noise=optimize(engine,initial,seed,train_nll,args.smoke)
    name=f'four_s{seed}';histories[name]=history;new_weights[name]=weights
    current=load_anchored(config['paths']['control']);current.params=residual_parameters(initial,embed_weights(weights))
    check_subspace(initial,current.params,weights);save_anchored(out/f'{name}.pkl',current,config['iterations'])
    models[name]=load_anchored(out/f'{name}.pkl');np.save(out/f'{name}_perturbations.npy',noise)
    write_json(out/'optimization_history.json',histories)
  reset_metrics={};continuation_metrics={};sources={};returns={};freeze={}
  old_reset=read(PRIOR/'reset_evaluation.json');old_cont=read(PRIOR/'continuation_evaluation.json')
  for name,current in models.items():
    if name in OLD:
      sources[name]={'reset':str(PRIOR/f'{OLD[name]}_trajectories.npz'),'continuation':str(PRIOR/f'{OLD[name]}_continuation_trajectories.npz'),
                     'probe':str(PROBES/f'{PROBE_NAMES[name]}_model_probes.npz'),'checkpoint':str(PRIOR/f'{OLD[name]}.pkl')}
      reset_metrics[name]=old_reset['models'][OLD[name]];continuation_metrics[name]=old_cont['models'][OLD[name]]
    else:
      current_engine=TaskRollout(current,nominal,actor)
      reset_parts=[current_engine.run(np.zeros(6,np.float32),START[None],GOAL[None],jax.random.PRNGKey(k),128) for k in config['evaluation_keys']]
      reset=concatenate_rollouts(reset_parts);save_rollouts(out,name,reset)
      final_nll=float(-_log_prob(current.diagonal,validation,4096).mean())
      if final_nll!=nll:raise RuntimeError('held-out diagonal changed')
      reset_metrics[name]={**rollout_metrics(reset),'heldout_diagonal_nll':final_nll,
          'per_evaluation_seed_return':[float(r['return'].mean()) for r in reset_parts],
          'checkpoint_sha256':file_sha(out/f'{name}.pkl')}
      record=current_engine.run(np.zeros(6,np.float32),contexts.state,contexts.goal,jax.random.PRNGKey(720001),4,50-contexts.timestep)
      save_rollouts(out,name+'_continuation',record);continuation_metrics[name]={}
      for group in np.unique(groups):
        part={k:v[groups==group] for k,v in record.items()}
        continuation_metrics[name][group]={**rollout_metrics(part),'mean_first_step_motion':float(np.linalg.norm(part['states'][:,:,1,:2]-part['states'][:,:,0,:2],axis=-1).mean())}
      check_subspace(initial,current.params,new_weights[name])
      freeze[name]={'exact_zero_bias':True,'only_selected_weights_changed':True,'weights':new_weights[name],
                    'diagonal_nll_unchanged':True,'checkpoint_sha256':file_sha(out/f'{name}.pkl')}
      sources[name]={'reset':str(out/f'{name}_trajectories.npz'),'continuation':str(out/f'{name}_continuation_trajectories.npz'),
                     'probe':str(out/f'{name}_model_probes.npz'),'checkpoint':str(out/f'{name}.pkl')}
    returns[name]=load_arrays(sources[name]['reset'])['return']
  selected=read(PROBES/'contexts.json');probe_state=np.repeat(np.array(selected['state'],np.float32),2,0)
  probe_goal=np.repeat(np.array(selected['commanded_goal'],np.float32),2,0)
  xp=np.array(selected['nominal_observational_actions'],np.float32).reshape(-1,2);probe_groups=np.repeat(selected['groups'],2)
  replay_xp=np.asarray(nominal.sample(np.array(selected['state'],np.float32),jax.random.PRNGKey(950000),2,goal=np.array(selected['commanded_goal'],np.float32))).reshape(-1,2)
  if not np.array_equal(xp,replay_xp):raise RuntimeError('nominal probe draws changed')
  probe_metrics={};probe_checks={}
  for name,current in models.items():
    record,checks=InstrumentedProbe(current).run(probe_state,xp,probe_goal,jax.random.PRNGKey(950001),64)
    if name in OLD:
      require_saved_equal(record,sources[name]['probe']);compatibility[name]['action_probe_reproduction_exact']=True
    else:np.savez_compressed(sources[name]['probe'],**record)
    probe_metrics[name]={'all':response_metrics(record),**{g:response_metrics({k:v[probe_groups==g] for k,v in record.items()}) for g in GROUPS}}
    probe_checks[name]=checks
  write_json(out/'action_response_metrics.json',probe_metrics);write_json(out/'probe_checks.json',probe_checks)
  write_json(out/'evaluation.json',{'reset':reset_metrics,'continuation':continuation_metrics,
      'paired_four_minus_control':{n:return_interval(v-returns['control']) for n,v in returns.items() if n.startswith('four')},
      'paired_four_minus_six':{n:return_interval(v-returns[n.replace('four','six')]) for n,v in returns.items() if n.startswith('four')},
      'continuation_contexts':{'episode_id':contexts.episode_id,'timestep':contexts.timestep,'groups':groups,'key':720001,'replicates':4,'horizon':'50 minus original timestep'},
      'action_context_file':str(PROBES/'contexts.json'),'action_sampling_key':950001,'artifact_sources':sources})
  if not tree_equal(nominal.params,nominal_copy) or not np.array_equal(actor_before,actor(START[None],GOAL[None],jax.random.PRNGKey(990))):
    raise RuntimeError('frozen policy changed')
  write_json(out/'freeze_checks.json',{'arms':freeze,'nominal_parameters_unchanged':True,'actor_fixed_key_output_unchanged':True,'original_model_unchanged':tree_equal(model.params,initial)})
  write_json(out/'compatibility.json',compatibility)
  config.update(status='complete',elapsed_seconds=time.time()-start,
      all_previous_inputs_unchanged=all(file_sha(p)==v for p,v in hashes.items()),
      source_sha256={p:file_sha(p) for p in ('ett/bias_ablation.py','ett/rollout_return.py','ett/action_response.py','scripts/test_bias_ablation.py')})
  if not config['all_previous_inputs_unchanged']:raise RuntimeError('prior inputs changed')
  write_json(out/'config.json',config);print('completed bias ablation in',config['elapsed_seconds'],'seconds',flush=True)


if __name__=='__main__':main()
