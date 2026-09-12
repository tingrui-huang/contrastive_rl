"""Samplewise action-Lipschitz ETT with an immutable diagonal sampling law.

The proof and limitations are in notes/ett_adversarial_fixed_actor_spec.md.
Only action-independent geometry choices precede nonexpansive box clipping.
No emitted likelihood is claimed for the mixed projected output measure.
"""
import jax
import jax.numpy as jnp
import numpy as np

from ett.diagonal_transition import sample_displacement, _project_samples, _assemble_inputs
from ett.rollout_return import task_reward, validate_fixed_goal


DIMENSION = 32
RECTANGLE_LOW = np.array([[0,3],[1,1],[1,1],[7,1]],np.float32)
RECTANGLE_HIGH = np.array([[9,4],[8,2],[2,4],[8,4]],np.float32)
# Floor-based static geometry excludes an interior rectangle's upper edge.
RECTANGLE_HIGH = np.where(RECTANGLE_HIGH==9,RECTANGLE_HIGH,
                          np.nextafter(RECTANGLE_HIGH,np.float32(-np.inf)))


def gates(state):
    motion=jnp.sqrt(jnp.mean(jnp.sum(jnp.diff(state.reshape(-1,4,2),axis=1)**2,axis=-1),axis=-1))
    x_centers=jnp.repeat(jnp.array([.5,2.5,4.5,7.5]),2)
    v_centers=jnp.tile(jnp.array([0.,.7]),4)
    logits=-.5*((state[:,0,None]-x_centers)/2)**2-.5*((motion[:,None]-v_centers)/.35)**2
    return jax.nn.softmax(logits,axis=-1)


def matrices(theta, bound=1.):
    raw=jnp.asarray(theta).reshape(8,2,2)
    norm=jnp.sqrt(jnp.sum(raw**2,axis=(1,2),keepdims=True))
    return bound*raw/jnp.maximum(1.,norm)


def convex_box(state, anchor):
    """Choose a rectangle per anchor draw, independent of execution action."""
    lo=jnp.maximum(jnp.asarray(RECTANGLE_LOW)[None,None],state[:,None,None,:2]-1.)
    hi=jnp.minimum(jnp.asarray(RECTANGLE_HIGH)[None,None],state[:,None,None,:2]+1.)
    lo=jnp.broadcast_to(lo,anchor.shape[:-1]+(4,2));hi=jnp.broadcast_to(hi,lo.shape)
    contains=jnp.all((anchor[...,None,:]>=lo)&(anchor[...,None,:]<=hi),axis=-1)
    area=jnp.prod(jnp.maximum(hi-lo,0),axis=-1)
    index=jnp.argmax(jnp.where(contains,area,-1.),axis=-1)
    take=lambda value:jnp.take_along_axis(value,index[...,None,None],axis=-2)[...,0,:]
    return take(lo),take(hi),jnp.any(contains,axis=-1),index


def emit(theta,state,action,xp,anchor,bound=1.):
    lo,hi,valid,rectangle=convex_box(state,anchor)
    matrix=jnp.einsum('bj,jkl->bkl',gates(state),matrices(theta,bound))
    response=jnp.einsum('bij,bj->bi',matrix,action-xp)
    proposal=anchor+response[:,None,:]
    xy=jnp.clip(proposal,lo,hi)
    # An uncovered anchor is a contract error, not a fallback law.
    xy=jnp.where(valid[...,None],xy,jnp.nan)
    old=jnp.broadcast_to(state[:,None,:6],xy.shape[:-1]+(6,))
    output=jnp.concatenate([xy,old],axis=-1)
    return output,dict(anchor_xy=anchor,box_low=lo,box_high=hi,box_valid=valid,rectangle=rectangle,
                       response=response,projection_corrected=jnp.any(xy!=proposal,axis=-1),
                       projected_to_boundary=jnp.any((xy==lo)|(xy==hi),axis=-1),
                       emitted_change=jnp.linalg.norm(xy-anchor,axis=-1))


class ConvexActionTransition:
    def __init__(self,diagonal,theta=None,bound=1.):
        if not np.isfinite(bound) or bound<=0:raise ValueError('positive finite bound required')
        self.diagonal=diagonal;self.bound=float(bound)
        self.theta=jnp.zeros(DIMENSION) if theta is None else jnp.asarray(theta,dtype=jnp.float32)
        if self.theta.shape!=(DIMENSION,) or not np.isfinite(self.theta).all():raise ValueError('expected 32 finite parameters')
        self._sample=jax.jit(self.sample_flat,static_argnums=(6,))

    def sample_flat(self,theta,state,action,xp,goal,key,count):
        base=self.diagonal
        context=jnp.concatenate([state,xp,xp,goal],axis=-1)
        distribution=base._distribution(base.params,context)
        delta,atom=sample_displacement(distribution,key,count,base.delta_mean,base.delta_std,base.spec)
        anchor,details=_project_samples(state,delta,base.spec)
        output,diagnostic=emit(theta,state,action,xp,anchor[...,:2],self.bound)
        diagnostic.update(stationary_atom=atom,
            base_corrected=jnp.any(details['raw_position']!=anchor[...,:2],axis=-1))
        return output,diagnostic

    def sample_with_diagnostics(self,state_or_observation,action,observational_action,key,num_samples=1,goal=None):
        if int(num_samples)!=num_samples or num_samples<1:raise ValueError('positive integer sample count required')
        state,context=_assemble_inputs(self.diagonal.spec,state_or_observation,action,observational_action,goal)
        leading=state.shape[:-1];flat=context.reshape(-1,20)
        if not np.isfinite(context).all() or np.any(np.abs(np.asarray(flat[:,8:12]))>1):
            raise ValueError('inputs must be finite and actions in [-1,1]')
        output,detail=self._sample(self.theta,state.reshape(-1,8),flat[:,8:10],flat[:,10:12],flat[:,12:],key,int(num_samples))
        if not np.asarray(detail['box_valid']).all():raise ValueError('diagonal anchor outside admissible geometry')
        if num_samples==1:output=output[:,0].reshape(leading+(8,))
        else:output=output.reshape(leading+(num_samples,8))
        # Diagnostics keep flat [batch,draw,...] axes, including one sample.
        return output,detail

    def sample(self,*args,**kwargs):return self.sample_with_diagnostics(*args,**kwargs)[0]

    def log_prob(self,*args,**kwargs):
        raise NotImplementedError('emitted projected mixed law has no implemented density; use diagonal energy fitting score')


def energy_score(samples,target):
    """Unbiased K-draw energy score in native XY units, including any atoms."""
    k=samples.shape[1]
    if k<2:raise ValueError('energy U-statistic requires at least two samples')
    first=jnp.linalg.norm(samples-target[:,None],axis=-1).mean(1)
    distances=jnp.linalg.norm(samples[:,:,None]-samples[:,None,:],axis=-1)
    return first-distances.sum((1,2))/(2*k*(k-1))


def objective(diagonal_score,returns,weight):
    return diagonal_score+weight*jnp.mean(returns)


class ConvexRollout:
    def __init__(self,model,nominal,actor,horizon=50,discount=.95):
        self.model=model;self.horizon=horizon
        def generate(theta,state,goal,key):
            def step(s,key):
                bk,ak,tk=jax.random.split(key,3)
                xp=nominal.sample(s,bk,1,goal=goal)
                action=actor(s,goal,ak)
                y,d=model.sample_flat(theta,s,action,xp,goal,tk,1)
                ns=y[:,0];reward=task_reward(ns,goal)
                return ns,dict(state=ns,action=action,reward=reward,aux_x_prime=xp,
                    anchor=d['anchor_xy'][:,0],projection_corrected=d['projection_corrected'][:,0],
                    boundary=d['projected_to_boundary'][:,0],valid=d['box_valid'][:,0],
                    base_corrected=d['base_corrected'][:,0],stationary_atom=d['stationary_atom'][:,0])
            _,records=jax.lax.scan(step,state,jax.random.split(key,horizon))
            records=jax.tree.map(lambda value:jnp.swapaxes(value,0,1),records)
            records['states']=jnp.concatenate([state[:,None],records.pop('state')],axis=1)
            records['return']=records['reward']@jnp.power(discount,jnp.arange(horizon))
            return records
        self.generate=jax.jit(generate)

    def run(self,theta,state,goal,key):
        validate_fixed_goal(goal)
        result=jax.tree.map(np.asarray,self.generate(jnp.asarray(theta),jnp.asarray(state),jnp.asarray(goal),key))
        if not result['valid'].all() or not np.isfinite(result['states']).all():raise ValueError('invalid rollout geometry')
        return result
