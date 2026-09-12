"""Evaluation-only reference, decomposition and bound audits for the bank task."""
import numpy as np
import torch

from scripts.synthetic_failure_reference import optimal_diagonal, optimal_response, quadrature, population_reference
from scripts.synthetic_lipschitz_response import exact_interval_diagnostics, pair_diagnostics
from scripts.synthetic_shared_response import error_metrics, DISTANCE_EDGES


@torch.no_grad()
def predict(model,pairs):
    pairs=np.asarray(pairs,np.float64)
    return np.concatenate([model(torch.from_numpy(a)).numpy() for a in np.array_split(pairs,max(1,int(np.ceil(len(pairs)/8192))))])


def evaluate(model,config,data):
    diag,off=data;pd,po=predict(model,diag),predict(model,off)
    d=predict(model,np.stack((off[:,1],off[:,1]),-1));distance=np.abs(off[:,0]-off[:,1])
    change=po-d;truth=optimal_response(off);error=po-truth
    failure=float(np.mean((po+3)**2));diag_mse=float(np.mean(pd**2))
    ref_d=optimal_diagonal(diag[:,1]);ref_total=float(np.mean(ref_d**2)+.1*np.mean((truth+3)**2))
    result=dict(diagonal=error_metrics(pd,1.),failure_mse=failure,
        common_lambda01_total=diag_mse+.1*failure,reference_empirical_total=ref_total,
        paired_empirical_objective_gap=diag_mse+.1*failure-ref_total,
        reference_error=error_metrics(error,1.),diagonal_reference_error=error_metrics(pd-ref_d,1.),
        action_change_error=error_metrics(change+distance,1.),mean_diagonal=float(pd.mean()),
        diagonal_min=float(pd.min()),diagonal_max=float(pd.max()),mean_action_change=float(change.mean()),
        nonpositive_action_change_fraction=float(np.mean(change<=config['fp_absolute_excess_tolerance'])),strata=[])
    shift_only=float(np.mean((d+3)**2));recentered=float(np.mean((change+3)**2))
    result['decomposition']=dict(zero_response_failure=9.,shift_only_failure=shift_only,
        actual_failure=failure,recentered_failure=recentered,
        shift_first_failure_reduction=9-shift_only,action_second_failure_reduction=shift_only-failure,
        order='shift first, then add learned action change; exact but order-dependent',
        diagnostic_only=True)
    for lo,hi in zip(DISTANCE_EDGES[:-1],DISTANCE_EDGES[1:]):
        mask=(distance>=lo)&(distance<hi)
        result['strata'].append(dict(low=float(lo),high=float(hi),**error_metrics(error[mask],1.),
            failure_mse=float(np.mean((po[mask]+3)**2)),mean_change=float(change[mask].mean()),
            action_change_rmse=float(np.sqrt(np.mean((change[mask]+distance[mask])**2)))))
    result['nominal_regions']={}
    for name,mask in [('central',np.abs(off[:,1])<=.8),('boundary',np.abs(off[:,1])>.8)]:
        result['nominal_regions'][name]=dict(**error_metrics(error[mask],1.),failure_mse=float(np.mean((po[mask]+3)**2)))
    axis=np.linspace(-1,1,config['grid_size']);xx,xp=np.meshgrid(axis,axis)
    pairs=np.stack((xx.ravel(),xp.ravel()),-1)
    grid=predict(model,pairs).reshape(xx.shape);grid_ref=optimal_response(pairs).reshape(xx.shape)
    result['grid_reference_error']=error_metrics(grid-grid_ref,1.)
    rng=np.random.default_rng(config['evaluation_seed']+1)
    triples=rng.uniform(-1,1,(config['eval_pairs']*2,3))
    sep=np.abs(triples[:,0]-triples[:,1]);triples=triples[sep>=config['min_separation']][:config['eval_pairs']]
    left,right=predict(model,triples[:,[0,2]]),predict(model,triples[:,[1,2]])
    sep=np.abs(triples[:,0]-triples[:,1]);keep=distance>=config['min_separation']
    diag_sep=np.broadcast_to(np.diff(axis)[None,:],grid[:,:-1].shape)
    bounds={}
    for name,a,b,delta in [('learned_anchor',po[keep],d[keep],distance[keep]),
        ('prescribed_c',po[keep],np.zeros(keep.sum()),distance[keep]),
        ('arbitrary_pairs',left,right,sep),('grid_adjacent',grid[:,1:],grid[:,:-1],diag_sep)]:
        bounds[name]=pair_diagnostics(a,b,delta,config['ratio_tolerance'],config['fp_absolute_excess_tolerance'])
    bounds['max_below_learned_anchor_envelope']=float(max(0.,np.max(d-distance-po)))
    bounds['max_below_zero_anchor_envelope']=float(max(0.,np.max(-distance-po)))
    result['bounds']=bounds
    anchors=np.array(config['slice_anchors']);fine=np.linspace(-.04,.04,401)
    sp=np.stack(np.broadcast_arrays(axis[None,:],anchors[:,None]),-1)
    cp=np.stack(np.broadcast_arrays(anchors[:,None]+fine,anchors[:,None]),-1)
    at_anchor=predict(model,np.stack((anchors,anchors),-1));sides=[]
    for eps in config['side_offsets']:
        lp=np.stack((anchors-eps,anchors),-1);rp=np.stack((anchors+eps,anchors),-1)
        lv,rv=predict(model,lp),predict(model,rp)
        sides.append(dict(offset=eps,left_slope=((at_anchor-lv)/eps).tolist(),right_slope=((rv-at_anchor)/eps).tolist()))
    result['near_diagonal']=dict(anchors=anchors.tolist(),diagonal=at_anchor.tolist(),sides=sides)
    result['intervals'],slope_arrays=exact_interval_diagnostics(model,axis)
    with torch.no_grad():raw=model.conditioner(torch.from_numpy(axis[:,None])).numpy()[:,1:]
    desired=np.where(np.arange(16)<8,1.,-1.)
    wrong=(raw*desired[None,:]<-1)&slope_arrays['interval_feasible']
    result['wrong_saturation']=dict(count=int(wrong.sum()),feasible_intervals=int(slope_arrays['interval_feasible'].sum()),
        interpretation='A zero direct clamp gradient is a local optimization obstacle, not a global representational impossibility.')
    result['quadrature128']=quadrature(model,128);result['quadrature256']=quadrature(model,256)
    result['quadrature_total_difference']=abs(result['quadrature128']['common_lambda01_total']-result['quadrature256']['common_lambda01_total'])
    result['population_reference']=population_reference()
    arrays=dict(diagonal_pairs=diag,diagonal_prediction=pd,off_pairs=off,off_prediction=po,
        off_anchor=d,action_change=change,optimal_prediction=truth,grid_axis=axis,grid=grid,grid_reference=grid_ref,
        triples=triples,arbitrary_left=left,arbitrary_right=right,slice_anchors=anchors,
        slice_prediction=predict(model,sp.reshape(-1,2)).reshape(4,-1),
        slice_reference=optimal_response(sp),close_offsets=fine,
        close_prediction=predict(model,cp.reshape(-1,2)).reshape(4,-1),close_reference=optimal_response(cp),**slope_arrays)
    assert all(np.isfinite(a).all() for a in arrays.values())
    return result,arrays
