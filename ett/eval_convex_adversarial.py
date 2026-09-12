"""Independent return, diagonal and action-constraint evaluation; no fitting."""
import argparse
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np

from ett.convex_action_transition import energy_score,matrices
from ett.run_convex_adversarial import CONFIG,verify,setup,reset_run
from ett.run_return_readout import read,write
from ett.eval_diagonal_transition import _open_endpoint
from scripts.make_swamp_f4_failure_bank import file_sha


def quantile(value):return np.quantile(value,[0,.5,.95,.99,1]).tolist()


def checks(states,previous):
    assert np.isfinite(states).all() and _open_endpoint(states[...,:2]).all()
    np.testing.assert_array_equal(states[...,2:],previous[...,:6])
    assert np.max(np.abs(states[...,:2]-previous[...,:2]))<=1+CONFIG['numerical_constraint_tolerance']


def rollout_metrics(record):
    states=record['states'];before,after=states[:,:-1],states[:,1:]
    checks(after,before)
    assert np.max(np.abs(record['action']))<=1 and np.max(np.abs(record['aux_x_prime']))<=1
    delta=after[...,:2]-before[...,:2];motion=np.linalg.norm(delta,axis=-1)
    segment=before[...,:2][...,None,:]+np.linspace(0,1,21)[None,None,:,None]*delta[...,None,:]
    stationary_history=np.all(before.reshape(*before.shape[:2],4,2)==before[...,:2][...,None,:],axis=(-1,-2))
    stationary_history[:,0]=False
    near=np.min(np.abs(after[...,:2]-np.round(after[...,:2])),axis=-1)<1e-4
    returns=record['return']
    return dict(mean_return=float(returns.mean()),return_std=float(returns.std()),return_quantiles=quantile(returns),
        zero_return_fraction=float(np.mean(returns==0)),nonzero_return_fraction=float(np.mean(returns>0)),
        stationary_step_fraction=float(np.mean(motion==0)),mean_motion=float(motion.mean()),
        mean_displacement=delta.mean((0,1)),projection_fraction=float(record['projection_corrected'].mean()),
        base_projection_fraction=float(record['base_corrected'].mean()),boundary_fraction=float(record['boundary'].mean()),
        near_integer_boundary_fraction=float(near.mean()),last10_frozen_fraction=float(np.mean(np.all(motion[:,-10:]==0,axis=1))),
        stationary_nonreset_history_steps=int(stationary_history.sum()),
        renewed_motion_on_stationary_history_fraction=float(np.mean(motion[stationary_history]>1e-5)) if stationary_history.any() else None,
        mean_motion_on_stationary_history=float(motion[stationary_history].mean()) if stationary_history.any() else None,
        xy_trajectory_spread=float(np.sqrt(np.var(states[...,:2],axis=0).sum(-1)).mean()),
        sampled_21point_segment_wall_fraction=float(np.mean(np.any(~_open_endpoint(segment),axis=-1))),
        endpoints_open=True,coordinate_cap=True,f4_exact=True,actions_bounded=True,
        interpretation='model generated; sampled straight-segment audit is supplementary, not simulator reachability')


def diagonal_audit(root,model,theta):
    data=dict(np.load(root/'fit_contexts.npz'));before=dict(np.load(root/'diagonal_before.npz'));report={}
    for number,name in enumerate(['train','validation']):
        s,a,g,y=[data[name+'_'+field] for field in ['state','action','goal','target']]
        samples,_=model._sample(jnp.asarray(theta),s,a,a,g,jax.random.PRNGKey(CONFIG['seed_bases']['fit_noise']+number),32)
        samples=np.asarray(samples);reference=before[name+'_samples']
        drift=np.linalg.norm(samples[...,:2]-reference[...,:2],axis=-1)
        score=np.asarray(energy_score(jnp.asarray(samples[...,:2]),jnp.asarray(y[:,:2])))
        score_delta=score-before[name+'_energy'];errors=np.linalg.norm(samples[...,:2].mean(1)-y[:,:2],axis=-1)
        assert float(drift.max())<=CONFIG['diagonal_max_sample_drift']
        assert float(score_delta.max())<=CONFIG['diagonal_max_energy_increase']
        checks(samples,np.broadcast_to(s[:,None],samples.shape))
        report[name]=dict(maximum_coupled_drift=float(drift.max()),energy_mean=float(score.mean()),
            maximum_energy_score_increase=float(score_delta.max()),energy_score_change_quantiles=quantile(score_delta),
            predictive_mean_error_quantiles=quantile(errors),passed=True)
    return report


def action_audit(root,model,nominal,theta):
    data=dict(np.load(root/'fit_contexts.npz'));s=data['validation_state'][:128];g=data['validation_goal'][:128]
    rng=np.random.default_rng(CONFIG['seed_bases']['action'])
    xp=np.asarray(nominal.sample(s,jax.random.PRNGKey(CONFIG['seed_bases']['action']),1,goal=g))
    key=jax.random.PRNGKey(CONFIG['seed_bases']['action']+1);calls=0;max_excess=0.;max_ratio=0.
    def draw(a):
        nonlocal calls
        y,d=model._sample(jnp.asarray(theta),s,a,xp,g,key,8);y=np.asarray(y);d=jax.tree.map(np.asarray,d);calls+=1
        checks(y,np.broadcast_to(s[:,None],y.shape))
        assert d['box_valid'].all() and np.all(y[...,:2]>=d['box_low']) and np.all(y[...,:2]<=d['box_high'])
        return y,d
    grid=np.array([(x,y) for x in [-1.,0.,1.] for y in [-1.,0.,1.]],np.float32)
    actions=np.concatenate([np.broadcast_to(grid[:,None],(9,128,2)),xp[None]],axis=0)
    output=[];details=[]
    for a in actions:
        y,d=draw(a);output.append(y);details.append(d)
    output=np.array(output)
    for d in details:np.testing.assert_array_equal(d['anchor_xy'],details[0]['anchor_xy'])
    np.testing.assert_array_equal(output[-1,...,:2],details[0]['anchor_xy'])
    def compare(a,b,y1,y2):
        nonlocal max_excess,max_ratio
        distance=np.linalg.norm(y1-y2,axis=-1);radius=np.linalg.norm(a-b,axis=-1)[:,None]
        excess=distance-radius
        max_excess=max(max_excess,float(excess.max()))
        max_ratio=max(max_ratio,float(np.max(np.divide(distance,radius,out=np.zeros_like(distance),where=radius>0))))
        assert excess.max()<=CONFIG['numerical_constraint_tolerance']
    for i in range(10):
        for j in range(i):compare(actions[i],actions[j],output[i],output[j])
    for _ in range(16):
        a=rng.uniform(-1,1,(128,2)).astype(np.float32);b=rng.uniform(-1,1,(128,2)).astype(np.float32)
        y1,d1=draw(a);y2,d2=draw(b);np.testing.assert_array_equal(d1['anchor_xy'],d2['anchor_xy']);compare(a,b,y1,y2)
    near={}
    for gap in [1e-3,1e-5]:
        a=rng.uniform(-.8,.8,(128,2)).astype(np.float32);b=a+gap
        y1,_=draw(a);y2,_=draw(b);compare(a,b,y1,y2)
        near[str(gap)]=float(np.max(np.linalg.norm(y1-y2,axis=-1)/np.linalg.norm(a-b,axis=-1)[:,None]))
    spread=np.sqrt(np.sum(np.var(output[:9,...,:2],axis=0),axis=-1))
    response=output[:9,...,:2]-details[0]['anchor_xy'][None]
    energy=float(np.sum(response**2));centered=float(np.sum((response-response.mean(0))**2))
    bound_matrices=np.asarray(matrices(theta))
    report=dict(maximum_positive_action_bound_excess=max_excess,maximum_sampled_ratio=max_ratio,
        near_pair_ratios=near,max_matrix_frobenius=float(np.linalg.norm(bound_matrices,axis=(1,2)).max()),
        max_matrix_operator_norm=float(np.linalg.svd(bound_matrices,compute_uv=False).max()),
        all_pair_guarantee='fixed-anchor independent convex-box projection, norm-bounded signed matrix; proof in SPEC.md',
        diagonal_sample_identity=True,box_valid=True,endpoints_open=True,f4_exact=True,coordinate_cap=True,
        projection_fraction=float(np.mean([d['projection_corrected'].mean() for d in details])),
        action_invariant_fraction=float(np.mean(spread<1e-6)),mean_grid_action_spread=float(spread.mean()),
        common_response_energy_fraction=1-centered/energy if energy>0 else None,
        right_minus_left_mean_norm=float(np.linalg.norm(output[7,...,:2]-output[1,...,:2],axis=-1).mean()),
        up_minus_down_mean_norm=float(np.linalg.norm(output[5,...,:2]-output[3,...,:2],axis=-1).mean()),
        single_step_samples=calls*128*8,numerical_audit_not_global_proof=True)
    return report,dict(actions=actions,nominal_action=xp,states=s,goals=g,outputs=output,anchor=details[0]['anchor_xy'])


def evaluate(root):
    verify(root);training=read(root/'training.json')
    if training['status']!='complete':raise ValueError('training did not complete')
    model,nominal,actor,engine,_=setup();metrics={};all_returns={};evaluation_steps=0;additional=training['diagonal_single_step_samples']
    for name,arm in training['arms'].items():
        checkpoint=root/'checkpoints'/f'{name}.npz';assert file_sha(checkpoint)==arm['checkpoint_sha256']
        with np.load(checkpoint) as z:theta=z['theta']
        records=[]
        for seed in CONFIG['evaluation_seeds']:
            records.append(reset_run(engine,theta,seed,128));evaluation_steps+=128*50
        record={key:np.concatenate([r[key] for r in records],axis=0) for key in records[0]}
        groups=[dict(seed=seed,mean_return=float(r['return'].mean())) for seed,r in zip(CONFIG['evaluation_seeds'],records)]
        metrics[name]=rollout_metrics(record);metrics[name]['evaluation_groups']=groups
        metrics[name]['diagonal']=diagonal_audit(root,model,theta);additional+=4096*32
        metrics[name]['action_audit'],action_samples=action_audit(root,model,nominal,theta)
        additional+=metrics[name]['action_audit']['single_step_samples']
        np.savez_compressed(root/f'{name}_action_audit.npz',**action_samples)
        auxiliary={key:record.pop(key) for key in ['aux_x_prime','anchor']}
        np.savez_compressed(root/f'{name}_auxiliary.npz',**auxiliary)
        np.savez_compressed(root/f'{name}_rollouts.npz',**record)
        all_returns[name]=record['return']
        # Repeat one exact saved evaluation group from a reloaded theta.
        replay=reset_run(engine,np.load(checkpoint)['theta'],CONFIG['evaluation_seeds'][0],128)
        for field in ['states','action','reward','return']:
            np.testing.assert_array_equal(replay[field],record[field][:128])
        additional+=128*50
        print(name,'return',metrics[name]['mean_return'],'max excess',metrics[name]['action_audit']['maximum_positive_action_bound_excess'],flush=True)
    paired={};rng=np.random.default_rng(CONFIG['seed_bases']['bootstrap'])
    # Stratified within four independent evaluation-key groups, paired paths.
    indices=np.concatenate([rng.integers(128,size=(500,128))+128*g for g in range(4)],axis=1)
    for seed in CONFIG['seeds']:
        delta=all_returns[f's{seed}_adversarial']-all_returns[f's{seed}_control']
        paired[str(seed)]=dict(adversarial_minus_control=float(delta.mean()),interval95=np.quantile(delta[indices].mean(1),[.025,.975]),
            decreases_in_every_group=bool(all(delta.reshape(4,128).mean(1)<0)),by_group=delta.reshape(4,128).mean(1))
    np.testing.assert_array_equal(all_returns['s0_control'],all_returns['s1_control'])
    primary=training['primary_model_transitions_so_far']+evaluation_steps
    assert primary==CONFIG['primary_transition_budget']
    # Reserve 10,000 one-step equivalents for the finite formula-only tests.
    assert additional+10000<=CONFIG['additional_transition_budget']
    success=all(v['interval95'][1]<0 for v in paired.values())
    verify(root)
    write(root/'metrics.json',metrics);write(root/'paired_returns.json',paired)
    write(root/'completion.json',dict(status='complete',success_within_declared_class=success,
        diagonal_budget_passed=True,constraint_checks_passed=True,primary_model_transitions=primary,
        additional_model_transitions=additional,formula_test_reserved_cap=10000,
        total_count_including_test_reserve=primary+additional+10000,
        no_simulator_steps=True,no_value_estimator=True,frozen_inputs_unchanged=True,
        replayed_paths=4*128,checkpoint_files_local=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',required=True)
    evaluate(Path(parser.parse_args().run_dir))
