"""Independent geometry-mode evaluation and fixed-input diagnostics, no fitting."""
import argparse
from pathlib import Path
import time
import traceback

import jax
import jax.numpy as jnp
import numpy as np

from ett.convex_action_transition import (ConvexActionTransition,ConvexRollout,
    load_convex_checkpoint,emit_with_geometry,matrices,validate_selected_set,fork_segment)
from ett.run_geometry_comparison import (CONFIG,Ledger,read,write,sha,verify,setup,
    rollout,diagonal,frozen_fingerprints,check_record,tree_sha)


def summary(record):
    check_record(record)
    before,after=record['states'][:,:-1],record['states'][:,1:]
    q=before[...,:2];delta=after[...,:2]-q
    motion=np.linalg.norm(delta,axis=-1)
    anchor_delta=record['anchor']-q
    stationary_anchor=np.all(anchor_delta==0,axis=-1)
    stationary_history=np.all(before.reshape(*before.shape[:2],4,2)==q[...,None,:],axis=(-1,-2))
    stationary_history[:,0]=False
    supported=(q[...,0]>=1)&(q[...,0]<2)&(q[...,1]>=3)&(q[...,1]<4)
    selected=record['geometry_kind']==1
    eligible=record['segment_eligible']
    fraction=record['segment_fraction'][selected]
    mask_stats=lambda mask:dict(count=int(mask.sum()),
        emitted_motion_mean=float(motion[mask].mean()) if mask.any() else None,
        renewed_motion_fraction=float(np.mean(motion[mask]>1e-5)) if mask.any() else None)
    result=dict(mean_return=float(record['return'].mean()),zero_return_fraction=float(np.mean(record['return']==0)),
        supported_fork_steps=int(supported.sum()),eligible_fork_steps=int(eligible.sum()),
        selected_segment_steps=int(selected.sum()),total_steps=int(motion.size),
        segment_selection_fraction=float(selected.mean()),eligibility_fraction=float(eligible.mean()),
        mean_motion=float(motion.mean()),motion_quantiles=np.quantile(motion,[0,.5,.95,.99,1]),
        exact_stationary_step_fraction=float(np.mean(motion==0)),
        last10_stationary_path_fraction=float(np.mean(np.all(motion[:,-10:]==0,axis=1))),
        geometry_projection_fraction=float(record['projection_corrected'].mean()),
        selected_set_boundary_fraction=float(record['boundary'].mean()),
        near_grid_boundary_fraction=float(np.mean(np.min(np.abs(after[...,:2]-np.round(after[...,:2])),axis=-1)<1e-4)),
        stationary_anchor=mask_stats(stationary_anchor),stationary_history=mask_stats(stationary_history),
        stationary_atom_fraction=float(record['stationary_atom'].mean()),
        segment_fraction=dict(count=len(fraction),anchor_mass=float(np.mean(fraction==0)) if len(fraction) else None,
            endpoint_mass=float(np.mean(fraction==1)) if len(fraction) else None,
            mean=float(fraction.mean()) if len(fraction) else None,
            quantiles=np.quantile(fraction,[0,.5,.95,1]) if len(fraction) else None),
        max_literal_coordinate_step=float(np.max(np.abs(after[...,:2].astype(float)-q.astype(float)))))
    for group,mask in [('all',np.ones(motion.shape,bool)),('supported_fork',supported),('eligible',eligible)]:
        result[group+'_response']=dict(count=int(mask.sum()))
        for name,value in [('anchor_displacement',anchor_delta),('matrix_correction',record['response']),('emitted_displacement',delta)]:
            result[group+'_response'][name]=dict(mean=value[mask].mean(0) if mask.any() else None,
                mean_norm=float(np.linalg.norm(value[mask],axis=-1).mean()) if mask.any() else None)
    # Endpoint geometry alone does not certify the native path between endpoints.
    from ett.eval_diagonal_transition import _open_endpoint
    lines=q[...,None,:]+np.linspace(0,1,21)[None,None,:,None]*delta[...,None,:]
    result['sampled_straight_line_wall_fraction']=float(np.mean(np.any(~_open_endpoint(lines),axis=-1)))
    return result


def fixed_components(root,ledger,baseline,nominal,models):
    data=dict(np.load(root/'fit_contexts.npz'));s=data['validation_state'][:64];g=data['validation_goal'][:64]
    xp=np.asarray(nominal.sample(s,jax.random.PRNGKey(65000100),1,goal=g))
    def draw():
        y,d=baseline._sample(baseline.theta,s,xp,xp,g,jax.random.PRNGKey(65000101),8)
        validate_selected_set(s,y,jax.tree.map(np.asarray,d))
        np.testing.assert_array_equal(y[...,:2],d['anchor_xy'])
        return dict(state=s,goal=g,x_prime=xp,anchor=d['anchor_xy'])
    panel=ledger.call('component_anchors','component_anchors',512,dict(key=65000101),draw)
    states=[np.repeat(s,8,axis=0)];anchors=[panel['anchor'].reshape(-1,1,2)]
    nominal_actions=[np.repeat(xp,8,axis=0)];actions=[np.repeat(data['validation_action'][:64],8,axis=0)]
    for seed in [0,1]:
        with np.load(f'artifacts/ett_convex_set_component/fork_segment_v1/s{seed}_component_outputs.npz') as z:
            mask=z['fork_descent'];states.append(z['state'][mask]);anchors.append(z['anchor'][mask])
            nominal_actions.append(z['x_prime'][mask]);actions.append(z['action'][mask])
    s=np.concatenate(states);a=np.concatenate(anchors);xp=np.concatenate(nominal_actions);x=np.concatenate(actions)
    # 512 fresh coupled anchors + 225 preserved fork anchors. All following calls
    # are deterministic component evaluations, never re-sampled or fed forward.
    groups=np.r_[np.zeros(512,int),np.ones(225,int)]
    grid=np.array([(u,v) for u in [-1.,0.,1.] for v in [-1.,0.,1.]]+[(.25,-.5),(.250001,-.499999)],np.float32)
    results={};saved=dict(state=s,anchor=a,x_prime=xp,action=x,group=groups)
    for name,model in models.items():
        normalized=np.asarray(matrices(model.theta))
        assert np.linalg.norm(normalized,axis=(1,2)).max()<=1+2e-6
        for mode in CONFIG['modes']:
            tag=name+'__emit_'+mode
            fn=jax.jit(lambda s,x,p,a:emit_with_geometry(model.theta,s,x,p,a,geometry_mode=mode))
            output,detail=jax.tree.map(np.asarray,fn(s,x,xp,a));validate_selected_set(s,output,detail)
            diag,dd=jax.tree.map(np.asarray,fn(s,xp,xp,a));np.testing.assert_array_equal(diag[...,:2],a)
            end,eligible=jax.tree.map(np.asarray,fork_segment(jnp.asarray(s),jnp.asarray(a)))
            saved[tag+'_output']=output;saved[tag+'_response']=detail['response']
            results[tag]=dict(trained_geometry=model.geometry_mode,evaluated_geometry=mode,
                max_matrix_frobenius=float(np.linalg.norm(normalized,axis=(1,2)).max()),
                max_matrix_operator=float(np.linalg.svd(normalized,compute_uv=False).max()),groups={})
            for label,mask in [('fresh_common',groups==0),('saved_fork',groups==1)]:
                results[tag]['groups'][label]=dict(contexts=int(mask.sum()),
                    mean_anchor_displacement=(a[mask,0]-s[mask,:2]).mean(0),
                    mean_correction=detail['response'][mask].mean(0),
                    mean_emitted_displacement=(output[mask,0,:2]-s[mask,:2]).mean(0),
                    below_Y3=int(np.sum(output[mask,0,1]<3)),
                    mean_output=output[mask,0,:2].mean(0),eligible=int(eligible[mask].sum()))
            # Vectorize action grid, keeping every conditioning value fixed.
            n,k=len(s),len(grid)
            ys,ds=jax.tree.map(np.asarray,fn(np.repeat(s,k,0),np.tile(grid,(n,1)),np.repeat(xp,k,0),np.repeat(a,k,0)))
            validate_selected_set(np.repeat(s,k,0),ys,ds)
            for field in ['anchor_xy','rectangle','box_low','box_high']+(['segment_end','geometry_kind'] if mode=='fork_segment' else []):
                arr=ds[field].reshape((n,k)+ds[field].shape[1:])
                np.testing.assert_array_equal(arr,np.repeat(arr[:,:1],k,axis=1))
            values=ys[:,0].reshape(n,k,8).astype(float)
            od=np.linalg.norm(values[:,:,None]-values[:,None,:],axis=-1)
            ad=np.linalg.norm(grid.astype(float)[:,None]-grid.astype(float)[None,:],axis=-1)
            excess=float(np.maximum(0,od-ad).max());assert excess<=2e-6
            results[tag]['max_positive_action_bound_excess']=excess
            results[tag]['grid_mean_action_spread']=float(np.sqrt(values[...,:2].var(1).sum(-1)).mean())
    np.savez_compressed(root/'fixed_component_outputs.npz',**saved)
    write(root/'fixed_component_metrics.json',results)
    return results


def evaluate(root):
    verify(root);training=read(root/'training.json');assert training['status']=='complete'
    ledger=Ledger(root);base,nominal,actor,_,_=setup()
    assert frozen_fingerprints(base,nominal,actor)==read(root/'frozen_before.json')
    models={'zero':base}
    for name,arm in training['models'].items():
        path=root/'checkpoints'/f'{name}.npz';assert sha(path)==arm['checkpoint_sha256']
        models[name]=load_convex_checkpoint(base.diagonal,path,require_geometry=True)
        assert models[name].geometry_mode==arm['mode']
    reference=dict(np.load(root/'diagonal_initial.npz'));all_records={};metrics={};diagonal_reports={}
    for name,model in models.items():
        engine=ConvexRollout(model,nominal,actor)
        groups=[]
        for group,key in enumerate(CONFIG['evaluation_seeds']):
            record=rollout(ledger,engine,model.theta,key,128,f'eval_{name}_g{group}','evaluation',True)
            groups.append(record)
        record={k:np.concatenate([g[k] for g in groups]) for k in groups[0]}
        record['evaluation_seed']=np.repeat(CONFIG['evaluation_seeds'],128)
        record['path_index']=np.tile(np.arange(128),4)
        all_records[name]=record
        np.savez_compressed(root/f'eval_{name}.npz',**record)
        metrics[name]=summary(record)
        metrics[name]['geometry_mode']=model.geometry_mode
        metrics[name]['by_group']=[float(g['return'].mean()) for g in groups]
        metrics[name]['L_diag']=training['diagonal_score']
        metrics[name]['weighted_J']=CONFIG['weight']*metrics[name]['mean_return']
        metrics[name]['total_objective']=metrics[name]['L_diag']+metrics[name]['weighted_J']
        if name!='zero':
            diagonal_reports[name],_=diagonal(ledger,model,name,reference)
            reload=load_convex_checkpoint(base.diagonal,root/'checkpoints'/f'{name}.npz',require_geometry=True)
            replay=rollout(ledger,ConvexRollout(reload,nominal,actor),reload.theta,64000000,128,
                           f'replay_{name}','replay',True)
            for field in groups[0]: np.testing.assert_array_equal(replay[field],groups[0][field])
        print(name,'independent return',metrics[name]['mean_return'],flush=True)
    rng=np.random.default_rng(66000100)
    indices=np.concatenate([rng.integers(128,size=(2000,128))+128*g for g in range(4)],axis=1)
    paired={};diffs={}
    for seed in [0,1]:
        for mode in CONFIG['modes']:
            name=f'{mode}_s{seed}';diffs[name+'_minus_zero']=all_records[name]['return']-all_records['zero']['return']
        diffs[f'fork_minus_rectangle_s{seed}']=all_records[f'fork_segment_s{seed}']['return']-all_records[f'rectangle_s{seed}']['return']
    for name,delta in diffs.items():
        paired[name]=dict(mean=float(delta.mean()),interval95=np.quantile(delta[indices].mean(1),[.025,.975]),
                         by_group=delta.reshape(4,128).mean(1))
    np.savez_compressed(root/'paired_episode_differences.npz',**diffs,bootstrap_indices=indices)
    write(root/'paired_returns.json',paired);write(root/'metrics.json',metrics);write(root/'diagonal_verification.json',diagonal_reports)
    fixed_components(root,ledger,base,nominal,{k:v for k,v in models.items() if k!='zero'})
    assert frozen_fingerprints(base,nominal,actor)==read(root/'frozen_before.json')
    verify(root)
    categories={}
    for entry in ledger.data['entries']: categories[entry['category']]=categories.get(entry['category'],0)+entry['transitions']
    assert ledger.data['charged']<=CONFIG['hard_cap']
    write(root/'completion.json',dict(status='complete',sampled_model_transitions=ledger.data['charged'],categories=categories,
          planned=CONFIG['planned_transitions'],cap=CONFIG['hard_cap'],failed_calls=[e for e in ledger.data['entries'] if e['status']!='complete'],
          frozen_components_unchanged=True,reload_exact=True,selected_set_checks=True,diagonal_exact=True,
          no_native_rollouts=True,no_downstream_training=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',type=Path,required=True)
    args=parser.parse_args()
    try: evaluate(args.run_dir)
    except BaseException:
        write(args.run_dir/f'failed_evaluation_{time.time_ns()}.json',dict(error=traceback.format_exc(),source_sha256=sha(__file__)))
        raise
