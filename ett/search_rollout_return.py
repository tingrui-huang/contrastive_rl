"""Fixed-budget paired black-box search; no neural backpropagation or critic."""
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from ett.anchored_transition import load_anchored,save_anchored
from ett.eval_diagonal_transition import _log_prob
from ett.rollout_return import (GOAL,START,HEAD_ROWS,DISCOUNT,TaskRollout,
    antithetic_gradient,descent_step,residual_parameters,two_loss_objective)
from ett.run_distribution_matching import write_json
from ett.run_rollout_return import (SOURCE,PREVIOUS_OBJECTIVE,read,tree_equal,
    rollout_metrics,save_rollouts,concatenate_rollouts)
from scripts.make_swamp_f4_failure_bank import file_sha


def return_interval(delta,seed=921,replicates=2000):
  delta=np.asarray(delta).ravel()
  draws=np.random.default_rng(seed).integers(len(delta),size=(replicates,len(delta)))
  return {'mean':float(delta.mean()),'ci95_paired_model_paths':np.quantile(delta[draws].mean(1),[.025,.975]),
      'paths':len(delta),'bootstrap_replicates':replicates,
      'scope':'independent simulated trajectories conditional on fixed reset, fitted models and policies; no causal/environment uncertainty'}


def continuation_contexts(dataset):
  prior=read(PREVIOUS_OBJECTIVE/'config.json')
  arrays=dataset.arrays('validation')
  lookup={(int(e),int(t)):i for i,(e,t) in enumerate(zip(arrays.episode_id,arrays.timestep))}
  ids=np.array([lookup[(e,t)] for e,t in zip(prior['evaluation_episode_ids'],prior['evaluation_timesteps'])])
  possible=arrays.take(ids)
  frozen=(possible.timestep>0)&np.all(possible.state.reshape(-1,4,2)==possible.state[:,None,:2],axis=(1,2))
  moving=np.linalg.norm(possible.state[:,:2]-possible.state[:,2:4],axis=-1)>.05
  chosen=[];groups=[]
  for name,mask in [('observed_frozen_nonreset',frozen),('observed_moving',moving)]:
    indices=np.flatnonzero(mask)
    _,first=np.unique(possible.episode_id[indices],return_index=True)
    selected=indices[first[:32]]
    chosen.extend(selected.tolist());groups.extend([name]*len(selected))
  return possible.take(np.array(chosen)),np.array(groups)


def search(out,args,model,nominal,actor,dataset):
  if not args.preflight_dir:raise ValueError('a completed preflight directory is required')
  preflight=Path(args.preflight_dir)
  previous=read(preflight/'config.json');gate=read(preflight/'preflight.json')
  if previous['status']!='complete' or not gate['search_signal_gate_passed']:
    raise ValueError('search blocked: frozen full-rollout return signal was not verified')
  for path,expected in previous['input_file_sha256'].items():
    if file_sha(path)!=expected:raise ValueError(f'preflight dependency changed: {path}')
  if args.iterations<1 or args.directions<1 or args.search_replicates<1:
    raise ValueError('positive fixed search budgets required')
  initial=model.params
  diagonal_copy=jax.tree.map(lambda v:np.array(v),initial['diagonal'])
  nominal_copy=jax.tree.map(lambda v:np.array(v),nominal.params)
  actor_probe=np.asarray(actor(START[None],GOAL[None],jax.random.PRNGKey(990)))
  nominal_probe=np.asarray(nominal.sample(START[None],jax.random.PRNGKey(991),8,goal=GOAL[None]))
  train=dataset.arrays('train');validation=dataset.arrays('validation')
  train_nll=float(-_log_prob(model.diagonal,train,4096).mean())
  validation_nll=float(-_log_prob(model.diagonal,validation,4096).mean())
  engine=TaskRollout(model,nominal,actor)
  sigma=.2;step_size=.1;max_step_norm=.2
  seeds=list(map(int,args.seeds.split(',')))
  budget={'objective':'L_diag + lambda * mean(sum_t gamma**t r_g(s_next)); exactly two terms',
      'scope':'restricted residual-head search; diagonal loss is constant, NOT joint two-loss training',
      'preflight_config_sha256':file_sha(preflight/'config.json'),
      'parameter_dimension':6,'parameter_definition':{'final_layer_bias_coordinates':2,'final_layer_weight_rows':HEAD_ROWS},
      'all_other_residual_and_diagonal_parameters_frozen':True,
      'noise':'independent N(0,I_6) directions; paired plus/minus; identical full-rollout keys within each pair and between lambda arms',
      'perturbation_sigma':sigma,'descent_learning_rate':step_size,'maximum_parameter_update_l2':max_step_norm,
      'update_clipping_is_not_a_loss':True,'lambda_values':[0.,1.],'optimizer_seeds':seeds,
      'iterations':args.iterations,'directions_per_iteration':args.directions,
      'paths_per_function_evaluation':args.search_replicates,'primary_initial_state':START,
      'horizon':50,'discount':DISCOUNT,'selection':'final fixed-budget checkpoint; no test-based selection',
      'parameter_query_rollouts_per_arm':args.iterations*args.directions*2*args.search_replicates,
      'center_monitor_rollouts_per_arm':(args.iterations+1)*args.search_replicates,
      'primary_evaluation_rollouts_per_arm':4*128,
      'diagonal_train_nll_constant':train_nll,'diagonal_validation_nll_constant':validation_nll,
      'diagonal_measure':'unprojected mixed zero atom / raw maze-unit XY area, nats per 2D transition',
      'estimator':'antithetic ES estimates the Gaussian-smoothed full-trajectory expectation, including hard reward, mixture decisions and geometry branches; no pathwise approximation',
      'no_hidden_reward_or_hindsight_goal':True,'source_sha256':file_sha(__file__)}
  write_json(out/'search_config.json',budget)
  final_theta={};histories={}
  for seed in seeds:
    directions=np.random.default_rng(600000+seed).normal(size=(args.iterations,args.directions,6)).astype(np.float32)
    for weight in (0.,1.):
      name=f's{seed}_lambda{int(weight)}';theta=np.zeros(6,np.float32)
      def center_score(value):
        # Fixed monitoring draws are never used for selection or directions.
        return float(engine.run(value,START[None],GOAL[None],jax.random.PRNGKey(620000+seed),args.search_replicates)['return'].mean())
      history=[{'iteration':0,'center_monitor_return':center_score(theta),'theta_norm':0.}]
      for iteration in range(args.iterations):
        plus=[];minus=[]
        for direction,epsilon in enumerate(directions[iteration]):
          key=jax.random.PRNGKey(610000+seed*10000+iteration*100+direction)
          positive=engine.run(theta+sigma*epsilon,START[None],GOAL[None],key,args.search_replicates)
          negative=engine.run(theta-sigma*epsilon,START[None],GOAL[None],key,args.search_replicates)
          rollout_metrics(positive);rollout_metrics(negative)
          plus.append(float(positive['return'].mean()));minus.append(float(negative['return'].mean()))
        gradient=antithetic_gradient(plus,minus,directions[iteration],sigma,weight)
        if not np.isfinite(gradient).all():raise RuntimeError('non-finite ES estimate')
        theta=descent_step(theta,gradient,step_size,max_step_norm)
        if weight==0 and np.any(theta!=0):raise RuntimeError('constant-loss control moved')
        monitor=center_score(theta)
        total,terms=two_loss_objective(train_nll,jnp.array([monitor]),weight)
        history.append({'iteration':iteration+1,'positive_returns':plus,'negative_returns':minus,
            'paired_return_differences':np.array(plus)-minus,'estimated_gradient':gradient,
            'gradient_norm':float(np.linalg.norm(gradient)),'theta_norm':float(np.linalg.norm(theta)),
            'center_monitor_return':monitor,'diagonal_loss':float(terms[0]),'off_loss':float(terms[1]),'two_term_loss':float(total)})
        if (iteration+1)%3==0:
          print(f'{name} iter={iteration+1} center return={monitor:.4f} theta norm={np.linalg.norm(theta):.3f}',flush=True)
      final_theta[name]=theta;histories[name]=history
      final=load_anchored(SOURCE/'s0_L0p25_lambda0/final.pkl')
      final.params=residual_parameters(initial,jnp.asarray(theta))
      if not tree_equal(final.params['diagonal'],diagonal_copy):raise RuntimeError('diagonal backbone moved')
      for key in ('mlp/~/linear_0','mlp/~/linear_1'):
        if not tree_equal(final.params['residual'][key],initial['residual'][key]):raise RuntimeError('residual torso moved')
      other_rows=np.setdiff1d(np.arange(64),HEAD_ROWS)
      if not np.array_equal(final.params['residual']['linear']['w'][other_rows],initial['residual']['linear']['w'][other_rows]):
        raise RuntimeError('parameter search escaped its six-dimensional subspace')
      save_anchored(out/f'{name}.pkl',final,args.iterations)
      write_json(out/'optimization_history.json',histories)
  # Evaluation keys are disjoint from directions, training rollouts and monitors.
  eval_keys=(710001,710002,710003,710004);metrics={};returns={}
  for name,theta in final_theta.items():
    records=[engine.run(theta,START[None],GOAL[None],jax.random.PRNGKey(seed),128) for seed in eval_keys]
    record=concatenate_rollouts(records)
    metrics[name]={**rollout_metrics(record),'per_evaluation_seed_return':[float(r['return'].mean()) for r in records],
        'independent_model_return_interval':return_interval(record['return']),
        'checkpoint_sha256':file_sha(out/f'{name}.pkl')}
    returns[name]=record['return']
    save_rollouts(out,name,record)
    # Check actual x=x_prime sampling equality with shared base randomness.
    arrays=validation.take(np.arange(64));key=jax.random.PRNGKey(993)
    expected=model.sample_flat(initial,arrays.state,arrays.action,arrays.action,arrays.goal,key,4)[0]
    final_params=residual_parameters(initial,jnp.asarray(theta))
    actual=model.sample_flat(final_params,arrays.state,arrays.action,arrays.action,arrays.goal,key,4)[0]
    if not np.array_equal(actual,expected):raise RuntimeError('diagonal sampling identity failed')
    final=load_anchored(out/f'{name}.pkl')
    if not tree_equal(final.params['diagonal'],diagonal_copy):raise RuntimeError('saved checkpoint changed diagonal')
    final_nll=float(-_log_prob(final.diagonal,validation,4096).mean())
    if final_nll!=validation_nll:raise RuntimeError('held-out diagonal likelihood changed')
    metrics[name]['heldout_diagonal_nll']=final_nll
    metrics[name]['exact_diagonal_sampling_identity']=True
    print(f'eval {name}: return={metrics[name]["model_return_mean"]:.4f}, nonzero={metrics[name]["positive_return_fraction"]:.3f}',flush=True)
  paired={f's{seed}':return_interval(returns[f's{seed}_lambda1']-returns[f's{seed}_lambda0']) for seed in seeds}
  write_json(out/'reset_evaluation.json',{'models':metrics,'paired_lambda1_minus_lambda0':paired,
      'evaluation_keys':eval_keys,'replicates_per_key':128,'scope':'model predictions only, never real-environment returns'})
  contexts,groups=continuation_contexts(dataset)
  continuation={}
  for name,theta in final_theta.items():
    record=engine.run(theta,contexts.state,contexts.goal,jax.random.PRNGKey(720001),4,remaining=50-contexts.timestep)
    continuation[name]={}
    for group in np.unique(groups):
      mask=groups==group
      part={key:value[mask] for key,value in record.items()}
      continuation[name][group]=rollout_metrics(part)
      continuation[name][group]['mean_first_step_motion']=float(np.linalg.norm(part['states'][:,:,1,:2]-part['states'][:,:,0,:2],axis=-1).mean())
    save_rollouts(out,name+'_continuation',record)
  write_json(out/'continuation_evaluation.json',{'models':continuation,'groups':groups,
      'episode_id':contexts.episode_id,'timestep':contexts.timestep,
      'scope':'previously reused development contexts; observable frozen/moving strata are not death labels',
      'horizon':'remaining original episode transitions only; inactive records are masked and do not count toward returns or checks'})
  if not tree_equal(nominal.params,nominal_copy) or not tree_equal(model.params['diagonal'],diagonal_copy):
    raise RuntimeError('a frozen model was modified')
  if not np.array_equal(actor_probe,actor(START[None],GOAL[None],jax.random.PRNGKey(990))):raise RuntimeError('actor changed')
  if not np.array_equal(nominal_probe,nominal.sample(START[None],jax.random.PRNGKey(991),8,goal=GOAL[None])):raise RuntimeError('nominal changed')
  write_json(out/'freeze_checks.json',{'diagonal_parameters_unchanged':True,'nominal_parameters_unchanged':True,
      'actor_fixed_key_outputs_unchanged':True,'nominal_fixed_key_outputs_unchanged':True,
      'residual_torso_and_unselected_head_weights_unchanged':True,'heldout_nll_unchanged':True,
      'diagonal_identity_exact':True,'first_loss_is_constant':True,'joint_two_loss_training':False})
