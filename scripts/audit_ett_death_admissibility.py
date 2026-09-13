"""Death representation gate: saved native validation and deterministic components.

Never calls a sampler, simulator, optimizer, policy or critic. Native flags are
used only in validation output; the generator receives visible conditions only.
"""
import argparse
from pathlib import Path
import hashlib
import json

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import expit

from ett.anchored_transition import load_anchored
from ett.convex_action_transition import (emit_with_geometry,convex_box,fork_segment,
    validate_selected_set,matrices)
from ett.diagonal_transition import _project_samples
from ett.rollout_return import GOAL

ROOT=Path('artifacts/ett_death_admissibility/frozen_f4_gate_v1')
PRIOR=Path('artifacts/ett_geometry_comparison/zero_s01_u16_v1')
ROUTES=Path('artifacts/ett_route_diagnostic/fixed_visible_s01_n128_v1')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    def convert(x):
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,np.generic):return x.item()
        raise TypeError(type(x))
    Path(path).write_text(json.dumps(value,indent=2,default=convert)+'\n',encoding='utf-8',newline='\n')


def conditions():
    native=np.load(ROUTES/'main_native_shortcut.npz')
    audit=np.load(ROUTES/'main_native_shortcut_native_audit.npz')
    np.testing.assert_array_equal(native['reset_seed'],audit['reset_seed'])
    e=next(int(i) for i in np.argsort(native['reset_seed']) if audit['failure'][i].any())
    fatal=int(np.flatnonzero(audit['failure'][e])[0])
    assert not audit['failure'][e,:fatal].any() and audit['failure'][e,fatal:].all()
    q=audit['true_xy'][e,fatal+1]
    np.testing.assert_array_equal(audit['true_xy'][e,fatal+1:],np.broadcast_to(q,audit['true_xy'][e,fatal+1:].shape))
    assert np.all(native['reward'][e,fatal:]==0)
    np.testing.assert_array_equal(native['states'][e,fatal+4],np.tile(q.astype(np.float32),4))
    rows=[];labels=[]
    for label,t in [('fatal_pre',fatal),('dead_age0',fatal+1),('dead_age3',fatal+4)]:
        for suffix in ['off_diagonal','diagonal']:
            action=native['action'][e,t].copy()
            rows.append(dict(id=label+'_'+suffix,state=native['states'][e,t],execution_action=action,
                nominal_action=np.zeros(2,np.float32) if suffix=='off_diagonal' else action.copy(),goal=GOAL))
            labels.append(dict(id=rows[-1]['id'],source='saved native validation only',
                reset_seed=int(native['reset_seed'][e]),state_index=t,
                hidden_dead_in_this_recorded_context=bool(audit['failure'][e,t-1]) if t else False,
                semantic_category='native fatal entry observed' if label=='fatal_pre' else 'native absorbing death observed',
                ett_admissible_death='not established; native hidden condition is not an ETT conditional target'))
    extra=[('swamp_stationary',[4.5,3.5],[0,0]),('swamp_action',[4.5,3.5],[1,0]),
           ('reset_control',[.5,3.5],[1,0]),('lower_control',[1.5,1.5],[1,1]),
           ('goal_control',[8.5,3.5],[-1,0]),('hazard_entry',[2.5,3.5],[1,0])]
    for name,xy,action in extra:
        rows.append(dict(id=name,state=np.tile(xy,4).astype(np.float32),execution_action=np.array(action,np.float32),
                         nominal_action=np.zeros(2,np.float32),goal=GOAL))
        labels.append(dict(id=name,source='geometry-specified condition; no hidden realization assigned',
            semantic_category='native death excluded next step for every clipped action' if name.endswith('control') else
                              'native death possible, hidden status unresolved',
            ett_admissible_death='not established'))
    validation=dict(reset_seed=int(native['reset_seed'][e]),episode_index=e,fatal_transition=fatal,
        death_position_true_precision=q,remaining_recorded_absorbing_transitions=49-fatal,
        ignored_actions_nonzero=int(np.sum(np.linalg.norm(native['action'][e,fatal+1:],axis=-1)>0)),
        exact_post_death_XY_and_zero_reward=True,death_age3_F4_equal=True,
        categories=labels,hidden_fields_never_passed_to_generator=True)
    return rows,validation


def run(out):
    out.mkdir(parents=True,exist_ok=True)
    provenance=json.loads((ROOT/'provenance.json').read_text())
    for path,h in provenance['input_sha256'].items():assert sha(path)==h,path
    assert sha(ROOT/'SPEC.md')==provenance['specification_sha256']
    rows,validation=conditions();write(out/'conditions.json',rows)
    write(out/'native_validation_only.json',validation)
    s=np.array([r['state'] for r in rows],np.float32)
    x=np.array([r['execution_action'] for r in rows],np.float32)
    xp=np.array([r['nominal_action'] for r in rows],np.float32)
    g=np.broadcast_to(GOAL,s.shape)
    # Only visible condition tensors enter the frozen network. No sampler call.
    p=json.loads((PRIOR/'provenance.json').read_text())
    base=load_anchored(p['paths']['control']).diagonal
    context=jnp.asarray(np.concatenate([s,xp,xp,g],axis=-1))
    distribution=base._distribution(base.params,context)
    logit,mixture,loc,scale=map(np.asarray,distribution)
    assert np.isfinite(logit).all() and np.isfinite(loc).all() and np.isfinite(scale).all()
    assert np.all(scale>0) and np.all(np.asarray(base.delta_std)>0)
    atom_real=expit(logit.astype(float));atom_float=np.asarray(jax.nn.sigmoid(distribution.stationary_logit))
    assert np.all((atom_float>0)&(atom_float<1))
    noise=np.array([(u,v) for u in [-1.,0.,1.] for v in [-1.,0.,1.]],np.float32)
    raw=(loc[:,:,None,:]+scale[:,:,None,:]*noise[None,None])*np.asarray(base.delta_std)+np.asarray(base.delta_mean)
    assert raw.shape==(12,3,9,2)
    raw=raw.reshape(12,27,2)
    anchor_f4,detail=_project_samples(jnp.asarray(s),jnp.asarray(raw),base.spec)
    anchor=np.asarray(anchor_f4[...,:2])
    lo,hi,valid,rectangle=jax.tree.map(np.asarray,convex_box(jnp.asarray(s),jnp.asarray(anchor)))
    assert valid.all()
    support_reports=[]
    for i,row in enumerate(rows):
        # Seek two distinct, unmodified open anchors: a finite check supporting
        # the analytic positive-density/non-atomic X argument, not an action search.
        untouched=np.all(np.asarray(detail['raw_position'])[i]==anchor[i],axis=-1)
        distinct=[]
        for j in np.flatnonzero(untouched):
            if not distinct or abs(float(anchor[i,j,0]-anchor[i,distinct[0],0]))>1e-4:
                distinct.append(int(j))
            if len(distinct)==2:break
        support_reports.append(dict(id=row['id'],stationary_logit=logit[i],
            stationary_atom_probability_float32=atom_float[i],stationary_atom_probability_float64=atom_real[i],
            probabilities_are_not_death_probabilities=True,gaussian_mixture_logits=mixture[i],
            means_native=loc[i]*np.asarray(base.delta_std)+np.asarray(base.delta_mean),
            scales_native=scale[i]*np.asarray(base.delta_std),
            unclipped_distinct_X_support_indices=distinct,
            unclipped_support_anchors=anchor[i,distinct],
            current_X_strictly_in_every_selected_box=bool(np.all((lo[i,:,0]<s[i,0])&(s[i,0]<hi[i,:,0]))),
            selected_rectangles=np.unique(rectangle[i]),
            deterministic_anchor_X_range=[float(anchor[i,:,0].min()),float(anchor[i,:,0].max())]))
    # Concrete native death-age rows must support the obstruction empirically.
    for i in [2,3,4,5,6,7]:
        assert len(support_reports[i]['unclipped_distinct_X_support_indices'])==2
        assert support_reports[i]['current_X_strictly_in_every_selected_box']
        assert np.all(rectangle[i]==0)
    coeff={'zero':np.zeros(32,np.float32)}
    for name in ['rectangle_s0','rectangle_s1','fork_segment_s0','fork_segment_s1']:
        with np.load(PRIOR/'checkpoints'/f'{name}.npz') as z:coeff[name]=z['theta']
    results={};outputs={}
    for name,theta in coeff.items():
        assert np.linalg.norm(np.asarray(matrices(theta)),axis=(1,2)).max()<=1+2e-6
        for mode in ['rectangle','fork_segment']:
            y,d=jax.tree.map(np.asarray,emit_with_geometry(theta,s,x,xp,anchor,geometry_mode=mode))
            validate_selected_set(s,y,d)
            diagonal,dd=emit_with_geometry(theta,s,xp,xp,anchor,geometry_mode=mode)
            np.testing.assert_array_equal(diagonal[...,:2],anchor)
            outputs[name+'__'+mode]=y
            # Fractions over a deterministic support panel are NOT probabilities.
            results[name+'__'+mode]=[dict(id=row['id'],moving_support_points=int(np.sum(np.any(y[i,:,:2]!=s[i,:2],axis=-1))),
                total_support_points=27,mean_panel_correction=d['response'][i],
                mean_panel_displacement=np.mean(y[i,:,:2]-s[i,:2],axis=0),
                max_panel_displacement=float(np.max(np.linalg.norm(y[i,:,:2]-s[i,:2],axis=-1))),
                not_a_probability_estimate=True) for i,row in enumerate(rows)]
    # Full all-action-pairs one-step hazard-entry map, expressly not a death model.
    index=11;ss=s[index:index+1];aa=ss[:,None,:2];pp=xp[index:index+1]
    theta=np.tile([1.,0.,0.,0.],8).astype(np.float32)
    action_grid=np.array([(u,v) for u in [-1.,0.,1.] for v in [-1.,0.,1.]],np.float32)
    ys=[]
    for action in action_grid:
        y,d=emit_with_geometry(theta,ss,action[None],pp,aa)
        validate_selected_set(ss,y,d);ys.append(np.asarray(y)[0,0,:2])
    ys=np.array(ys);dist=np.linalg.norm(ys[:,None]-ys[None,:],axis=-1)
    radius=np.linalg.norm(action_grid[:,None]-action_grid[None,:],axis=-1)
    assert np.max(dist-radius)<=2e-6
    np.testing.assert_array_equal(ys[7],[3.5,3.5])
    write(out/'hazard_endpoint_only.json',dict(state=ss[0],anchor=aa[0,0],x_prime=pp[0],theta=theta,
        action_grid=action_grid,outputs=ys,
        full_action_bound='rank-one diag(1,0) and fixed upper rectangle projection; exact diagonal on each anchor draw',
        support_of_stationary_anchor=float(atom_float[index]),
        interpretation='Admissible hazard landing for this anchor draw; NOT verified death or an absorbing model witness.'))
    np.savez_compressed(out/'component_arrays.npz',state=s,action=x,x_prime=xp,goal=g,
        gaussian_noise_points=noise,raw_delta=raw,anchors=anchor,box_low=lo,box_high=hi,
        selected_rectangle=rectangle,**outputs)
    write(out/'diagonal_support.json',support_reports)
    write(out/'current_coefficient_components.json',results)
    for path,h in provenance['input_sha256'].items():assert sha(path)==h,path
    write(out/'verification.json',dict(starting_commit=provenance['starting_commit'],
        gate='representation/admissibility not established; stop dependent stages',
        diagnostic_conditions=12,projected_gaussian_support_points=324,emitter_component_outputs=3240,
        deterministic_diagonal_identity_outputs=3240,endpoint_witness_action_outputs=9,
        new_native_transitions=0,new_stochastic_model_draws=0,optimizer_updates=0,
        objective_comparison='not run: no absorbing model witness certified',optimization='not run: gate 1 unmet',
        verified_model_death_rate=None,all_input_hashes_unchanged=True,
        hidden_information_used_for_validation_only=True,source_sha256=sha(__file__)))
    print('Completed 12-condition deterministic gate. No certified death model; dependent tests stopped.')
    for v in support_reports:print(v['id'],'atom=',float(v['stationary_atom_probability_float32']),'support pair=',v['unclipped_support_anchors'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out-dir',type=Path,default=ROOT)
    run(parser.parse_args().out_dir)
