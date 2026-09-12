"""Evaluation-only execution-action probes; no loss, optimizer or model update."""
import jax
import jax.numpy as jnp
import numpy as np

from ett.diagonal_transition import _project_samples
from ett.eval_diagonal_transition import _open_endpoint
from ett.rollout_return import GOAL,validate_fixed_goal


ACTION_NAMES=('zero','right_half','left_half','up_half','down_half',
              'right_full','left_full','up_full','down_full','observational')
CARDINAL_ACTIONS=np.array([[0,0],[.5,0],[-.5,0],[0,.5],[0,-.5],
                          [1,0],[-1,0],[0,1],[0,-1]],np.float32)
GROUPS=('open_corridor','boundary','before_swamp','stationary_nonreset')
SELECTION_RULES={
    'open_corridor':'6.2 <= x <= 8.2, 3.2 < y < 3.8, history not fully stationary',
    'boundary':'distance to the edge of any static blocked grid cell or map boundary <= 0.06; nonstationary history',
    'before_swamp':'2 <= x < 3, 3 <= y < 4; nonstationary history',
    'stationary_nonreset':'t > 0 and all four observed XY frames exactly equal; no hidden-state interpretation',
    'sampling':'seeded row permutation, first eligible row from each distinct episode within each group; same episode may occur in different groups'}


def wall_distance(xy):
  from ett.diagonal_transition import POINTMAZE_WALLS
  cells=np.argwhere(POINTMAZE_WALLS==1)
  separation=np.maximum(np.maximum(cells[None]-xy[:,None],xy[:,None]-(cells+1)[None]),0)
  distance=np.linalg.norm(separation,axis=-1).min(-1)
  outer=np.minimum(xy,np.array([9.,5.])-xy).min(-1)
  return np.minimum(distance,outer)


def select_contexts(state,episode,timestep,count,seed):
  xy=state[:,:2]
  stationary=np.all(state.reshape(-1,4,2)==state[:,None,:2],axis=(1,2))
  masks=((xy[:,0]>=6.2)&(xy[:,0]<=8.2)&(xy[:,1]>3.2)&(xy[:,1]<3.8)&~stationary,
         (wall_distance(xy)<=.06)&~stationary,
         (xy[:,0]>=2)&(xy[:,0]<3)&(xy[:,1]>=3)&(xy[:,1]<4)&~stationary,
         (timestep>0)&stationary)
  order=np.random.default_rng(seed).permutation(len(state));chosen=[];groups=[];counts={}
  for name,mask in zip(GROUPS,masks):
    candidates=order[mask[order]];seen=set();selected=[]
    for index in candidates:
      if int(episode[index]) not in seen:
        selected.append(int(index));seen.add(int(episode[index]))
      if len(selected)==count:break
    counts[name]={'eligible_rows':int(mask.sum()),'eligible_episodes':len(np.unique(episode[mask])),
                  'selected':len(selected)}
    chosen.extend(selected);groups.extend([name]*len(selected))
  return np.array(chosen,dtype=int),np.array(groups),counts


def actions_for(observational):
  xp=np.asarray(observational,np.float32)
  if not np.isfinite(xp).all() or np.any(np.abs(xp)>1):raise ValueError('invalid observational action; no silent clipping')
  base=np.broadcast_to(CARDINAL_ACTIONS,(len(xp),9,2))
  return np.concatenate([base,xp[:,None]],axis=1)


class InstrumentedProbe:
  """Read the unmodified sampler, then reconstruct intermediate geometry.

  Instrumentation never replaces the returned next state. A second ordinary
  fixed-key call verifies exact equality; reconstructed internal values have
  an explicitly checked float32 tolerance because JIT fusion can differ.
  """
  def __init__(self,model):
    self.model=model
    def reconstruct(state,action,xp,goal,anchor):
      base=model.diagonal;count=anchor.shape[1]
      context=(jnp.concatenate([state,action,xp,goal],-1)-base.context_mean)/base.context_std
      expanded=jnp.broadcast_to(context[:,None],(len(state),count,20))
      difference=action-xp
      inputs=jnp.concatenate([expanded,jnp.broadcast_to(difference[:,None],(len(state),count,2)),anchor-state[:,None,:2]],-1)
      raw=model.residual.apply(model.params['residual'],inputs.reshape(-1,24)).reshape(len(state),count,2)
      direction=jnp.tanh(raw)/jnp.sqrt(2.)
      radius=model.action_bound*jnp.linalg.norm(difference,axis=-1)
      margin=8*jnp.finfo(anchor.dtype).eps*(1+jnp.max(jnp.abs(anchor),axis=-1)+radius[:,None])
      safe_radius=jnp.maximum(radius[:,None]-margin,0.)
      proposal=anchor+safe_radius[...,None]*direction
      candidate,detail=_project_samples(state,proposal-state[:,None,:2],base.spec)
      distance=jnp.linalg.norm(candidate[...,:2]-anchor,axis=-1)
      fallback=(distance>radius[:,None])|(radius[:,None]==0)
      emitted=jnp.where(fallback[...,None],anchor,candidate[...,:2])
      return {'raw_network':raw,'direction':direction,'safe_radius':safe_radius,
          'proposal_xy':proposal,'projected_xy':candidate[...,:2],
          'projection_delta':candidate[...,:2]-proposal,'fallback_delta':emitted-candidate[...,:2],
          'reconstructed_emitted_xy':emitted,'pre_projection_blocked':detail['blocked_endpoint_before_projection']}
    self._reconstruct=jax.jit(reconstruct)

  def run(self,state,xp,goal,key,count):
    validate_fixed_goal(goal)
    actions=actions_for(xp);records=[];max_error=0.
    for i in range(len(ACTION_NAMES)):
      action=actions[:,i]
      output,detail=self.model.sample_with_diagnostics(state,action,xp,key,count,goal)
      original=self.model.sample(state,action,xp,key,count,goal)
      if not np.array_equal(output,original):raise RuntimeError('instrumentation altered sampler output')
      record={name:np.asarray(value) for name,value in detail.items()}
      record.update({name:np.asarray(value) for name,value in self._reconstruct(
          jnp.asarray(state),jnp.asarray(action),jnp.asarray(xp),jnp.asarray(goal),detail['anchor_xy']).items()})
      record['next_state']=np.asarray(output)
      error=float(np.max(np.abs(record['reconstructed_emitted_xy']-record['next_state'][...,:2])))
      if error>2e-6:raise RuntimeError(f'intermediate reconstruction mismatch: {error}')
      max_error=max(max_error,error);records.append(record)
    result={name:np.stack([r[name] for r in records],axis=1) for name in records[0]}
    result.update(state=np.asarray(state),goal=np.asarray(goal),observational_action=xp,execution_action=actions)
    if not np.all(result['anchor_xy']==result['anchor_xy'][:,:1]):raise RuntimeError('anchor depends on execution action')
    expected=np.broadcast_to(state[:,None,None,:6],result['next_state'][...,2:].shape)
    if not np.array_equal(result['next_state'][...,2:],expected):raise RuntimeError('F4 shift failed')
    if not np.array_equal(result['next_state'][:,-1,:,:2],result['anchor_xy'][:,-1]):raise RuntimeError('diagonal branch differs')
    if not _open_endpoint(result['next_state'][...,:2]).all():raise RuntimeError('invalid endpoint')
    if np.any(np.abs(result['next_state'][...,:2]-state[:,None,None,:2])>1+1e-6):raise RuntimeError('coordinate cap violated')
    if np.any(result['emitted_change']>result['radius']+1e-7):raise RuntimeError('anchor bound violated')
    if not all(np.isfinite(v).all() for v in result.values()):raise RuntimeError('non-finite probe')
    return result,{'unmodified_sampler_fixed_key_equality':True,'maximum_reconstruction_error':max_error,
                   'anchor_invariant_to_action':True,'exact_diagonal_branch':True,'exact_f4_shift':True,
                   'valid_endpoints_actions_and_anchor_bounds':True,'execution_action_clipped_count':0}


def response_metrics(record):
  # Nine fixed actions, excluding the identity action whose radius is zero.
  raw=record['proposal_xy'][:,:9];emitted=record['next_state'][:,:9,:,:2]
  direction=record['direction'][:,:9];radius=record['radius'][:,:9]
  valid=radius>1e-6
  weights=valid[...,None]
  mean=(direction*weights).sum(1)/np.maximum(weights.sum(1),1)
  total=float((direction**2*weights).sum())
  centered=float(((direction-mean[:,None])**2*weights).sum())
  def spread(v):
    difference=np.asarray(v,np.float64)-np.asarray(v[:,:1],np.float64)
    return float(np.mean(np.sum((difference-difference.mean(1,keepdims=True))**2,axis=-1)))
  correction=emitted-raw
  rawvar=spread(raw);emittedvar=spread(emitted);correctionvar=spread(correction)
  gain_x=(emitted[:,1]-emitted[:,2]).mean((0,1))
  gain_y=(emitted[:,3]-emitted[:,4]).mean((0,1))
  return {'contexts_times_observational_draws':len(raw),
      'common_direction_energy_fraction':None if total<1e-20 else 1-centered/total,
      'mean_common_direction':mean.mean((0,1)),
      'mean_raw_residual_magnitude':float(np.linalg.norm(raw-record['anchor_xy'][:,:9],axis=-1).mean()),
      'mean_emitted_anchor_displacement':float(record['emitted_change'][:,:9].mean()),
      'raw_action_variance':rawvar,'emitted_action_variance':emittedvar,
      'geometry_action_variance':correctionvar,
      'geometry_to_emitted_action_variance_ratio':None if emittedvar<1e-20 else correctionvar/emittedvar,
      'geometry_variance_note':'ratio is not an additive attribution fraction; raw and correction responses have a cross term',
      'candidate_corrected_fraction':float(record['candidate_corrected'][:,:9].mean()),
      'bound_fallback_fraction':float(record['bound_fallback'][:,:9].mean()),
      'mean_projection_correction':float(np.linalg.norm(record['projection_delta'][:,:9],axis=-1).mean()),
      'mean_fallback_correction':float(np.linalg.norm(record['fallback_delta'][:,:9],axis=-1).mean()),
      'half_action_central_response_columns_xy':np.stack([gain_x,gain_y],-1),
      'max_fixed_key_execution_action_effect':float(np.abs(emitted-emitted[:,:1]).max()),
      'half_opposing_x_response_norm':float(np.linalg.norm(emitted[:,1]-emitted[:,2],axis=-1).mean()),
      'half_opposing_y_response_norm':float(np.linalg.norm(emitted[:,3]-emitted[:,4],axis=-1).mean())}
