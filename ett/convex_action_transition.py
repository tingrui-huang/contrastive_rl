"""Samplewise action-Lipschitz ETT with an immutable diagonal sampling law.

The proof and limitations are in notes/ett_adversarial_fixed_actor_spec.md.
Only action-independent geometry choices precede convex projection. The optional
fork segment is specified in artifacts/ett_convex_set_component/fork_segment_v1/SPEC.md.
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


GEOMETRY_MODES = ('rectangle', 'fork_segment')


def fork_segment(state, anchor):
    """Select [anchor, end] from visible XY and anchor alone, or fall back.

    Arrays have [batch, draw, ...] axes. Bounds retain the existing float32
    rounded coordinate-step contract. No action, waypoint or label is accepted.
    """
    q = state[:, None, :2]
    step_low, step_high = q - 1., q + 1.
    valid = convex_box(state, anchor)[2]
    ax, ay = anchor[..., 0], anchor[..., 1]
    left = jnp.maximum(1., step_low[..., 0])
    bottom = jnp.maximum(1., step_low[..., 1])
    right = float(RECTANGLE_HIGH[2, 0])
    vertical = ax <= right
    height, dx = ay - 3., ax - 1.875
    # Safe denominators also keep masked, unsupported branches finite.
    extension = jnp.minimum((3. - bottom) / (2. * jnp.where(height > 0, height, 1.)),
                            (1.875 - left) / (2. * jnp.where(dx > 0, dx, 1.)))
    ray_end = jnp.stack([1.875 - extension * dx, 3. - extension * height], axis=-1)
    vertical_end = jnp.stack([ax, jnp.broadcast_to((3. + bottom) / 2., ax.shape)], axis=-1)
    end = jnp.where(vertical[..., None], vertical_end, ray_end)
    ex, ey = end[..., 0], end[..., 1]
    # A conservative certificate on the rounded endpoints, with wall clearance.
    crossing_ok = ((ax - 1.9375) * (3. - ey) <= (1.9375 - ex) * height)
    supported = ((q[..., 0] >= 1.) & (q[..., 0] < 2.) &
                 (q[..., 1] >= 3.) & (q[..., 1] < 4.))
    available = (valid & supported & (ax >= 1.) & (ay >= 3.) &
                 (ay <= float(RECTANGLE_HIGH[0, 1])) &
                 jnp.all((anchor >= step_low) & (anchor <= step_high), axis=-1) &
                 jnp.all(jnp.isfinite(end) & (end >= step_low) & (end <= step_high), axis=-1) &
                 (ex >= left) & (ex <= right) & (ey >= bottom) & (ey < 3.) &
                 (vertical | ((height > 0.) & crossing_ok)) &
                 (jnp.sum((end - anchor)**2, axis=-1) > 1e-12))
    return jnp.where(available[..., None], end, anchor), available


def project_segment(proposal, start, end):
    """Euclidean projection onto one fixed closed segment (including a point)."""
    direction = end - start
    squared_length = jnp.sum(direction**2, axis=-1)
    fraction = jnp.clip(jnp.sum((proposal - start) * direction, axis=-1) /
                        jnp.where(squared_length > 0., squared_length, 1.), 0., 1.)
    xy = start + fraction[..., None] * direction
    # Exact stored endpoints, notably the exact samplewise diagonal anchor.
    xy = jnp.where((fraction <= 0.)[..., None], start, xy)
    xy = jnp.where((fraction >= 1.)[..., None], end, xy)
    return xy, fraction


def emit_with_geometry(theta, state, action, xp, anchor, bound=1., *, geometry_mode='rectangle'):
    """Optional fixed convex-set projection; historical ``emit`` is unchanged.

    In segment mode box_* and rectangle describe the reference/fallback box,
    not a membership constraint on segment outputs. geometry_kind is 0 for the
    original box, 1 for a segment. Boundary means relative endpoints for a segment.
    """
    if geometry_mode not in GEOMETRY_MODES:
        raise ValueError(f'geometry_mode must be one of {GEOMETRY_MODES}')
    output, diagnostic = emit(theta, state, action, xp, anchor, bound)
    if geometry_mode == 'rectangle':
        return output, diagnostic
    end, available = fork_segment(state, anchor)
    proposal = anchor + diagnostic['response'][:, None, :]
    projected, fraction = project_segment(proposal, anchor, end)
    xy = jnp.where(available[..., None], projected, output[..., :2])
    output = jnp.concatenate([xy, output[..., 2:]], axis=-1)
    diagnostic.update(
        geometry_kind=available.astype(jnp.int32), segment_available=available,
        segment_start=anchor, segment_end=end, segment_fraction=fraction,
        projection_corrected=jnp.any(xy != proposal, axis=-1),
        projected_to_boundary=jnp.where(available, (fraction <= 0.) | (fraction >= 1.),
                                         diagnostic['projected_to_boundary']),
        emitted_change=jnp.linalg.norm(xy - anchor, axis=-1))
    return output, diagnostic


class ConvexActionTransition:
    def __init__(self,diagonal,theta=None,bound=1.,*,geometry_mode='rectangle'):
        if not np.isfinite(bound) or bound<=0:raise ValueError('positive finite bound required')
        if geometry_mode not in GEOMETRY_MODES:raise ValueError(f'geometry_mode must be one of {GEOMETRY_MODES}')
        self._geometry_mode=geometry_mode
        self.diagonal=diagonal;self.bound=float(bound)
        self.theta=jnp.zeros(DIMENSION) if theta is None else jnp.asarray(theta,dtype=jnp.float32)
        if self.theta.shape!=(DIMENSION,) or not np.isfinite(self.theta).all():raise ValueError('expected 32 finite parameters')
        self._sample=jax.jit(self.sample_flat,static_argnums=(6,))

    @property
    def geometry_mode(self):
        """Construction-time configuration; construct a new instance to change it."""
        return self._geometry_mode

    def sample_flat(self,theta,state,action,xp,goal,key,count):
        base=self.diagonal
        context=jnp.concatenate([state,xp,xp,goal],axis=-1)
        distribution=base._distribution(base.params,context)
        delta,atom=sample_displacement(distribution,key,count,base.delta_mean,base.delta_std,base.spec)
        anchor,details=_project_samples(state,delta,base.spec)
        output,diagnostic=emit_with_geometry(theta,state,action,xp,anchor[...,:2],self.bound,
                                             geometry_mode=self.geometry_mode)
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


def save_convex_checkpoint(path, model, theta):
    """Write a new coefficient checkpoint with mandatory geometry metadata."""
    theta = np.asarray(theta)
    if theta.shape != (DIMENSION,) or not np.isfinite(theta).all():
        raise ValueError('expected 32 finite parameters')
    with open(path, 'xb') as stream:
        np.savez(stream, theta=theta, bound=model.bound, geometry_mode=model.geometry_mode,
                 format_version=1)


def load_convex_checkpoint(diagonal, path, *, require_geometry=False):
    """Legacy theta/bound files mean rectangle; new files must retain their mode.

    New experiment evaluators use require_geometry=True so metadata loss cannot
    silently turn a newly produced checkpoint into a historical rectangle file.
    """
    with np.load(path, allow_pickle=False) as data:
        version = int(data['format_version']) if 'format_version' in data else None
        if version not in (None, 1):
            raise ValueError('unsupported convex checkpoint version')
        if (require_geometry or version is not None) and 'geometry_mode' not in data:
            raise ValueError('geometry mode missing from new coefficient checkpoint')
        mode = str(data['geometry_mode'].item()) if 'geometry_mode' in data else 'rectangle'
        return ConvexActionTransition(diagonal, data['theta'], float(data['bound']), geometry_mode=mode)


def validate_selected_set(state, output, diagnostic, tolerance=2e-6):
    """Validate actual per-draw convex sets, free endpoints, rounded steps and F4.

    state is [batch,8], output [batch,draw,8]; rectangle-only diagnostics remain
    accepted. This is a numerical component check, not a full-domain proof.
    """
    state, output = np.asarray(state), np.asarray(output)
    xy = output[..., :2]
    if not np.isfinite(output).all() or not np.asarray(diagnostic['box_valid']).all():
        raise ValueError('nonfinite output or invalid anchor box')
    np.testing.assert_array_equal(output[..., 2:], np.broadcast_to(state[:, None, :6], output[..., 2:].shape))
    low, high = state[:, None, :2]-1., state[:, None, :2]+1.
    assert np.all(xy >= low-tolerance) and np.all(xy <= high+tolerance)
    free = np.any(np.all((xy[..., None, :] >= RECTANGLE_LOW-tolerance) &
                         (xy[..., None, :] <= RECTANGLE_HIGH+tolerance), axis=-1), axis=-1)
    assert free.all()
    kind = np.asarray(diagnostic.get('geometry_kind', np.zeros(xy.shape[:-1], np.int32)))
    assert np.all((kind == 0) | (kind == 1))
    box = kind == 0
    assert np.all(xy[box] >= np.asarray(diagnostic['box_low'])[box]-tolerance)
    assert np.all(xy[box] <= np.asarray(diagnostic['box_high'])[box]+tolerance)
    segment = ~box
    if segment.any():
        a, b = [np.asarray(diagnostic[k]) for k in ['segment_start', 'segment_end']]
        fraction = np.asarray(diagnostic['segment_fraction'])
        assert np.all((fraction[segment] >= 0) & (fraction[segment] <= 1))
        projected = a + fraction[..., None]*(b-a)
        np.testing.assert_allclose(xy[segment], projected[segment], atol=tolerance, rtol=0)
    return True


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
                endpoint, eligible = fork_segment(s, d['anchor_xy'])
                kind = d.get('geometry_kind', jnp.zeros_like(d['rectangle']))
                return ns,dict(state=ns,action=action,reward=reward,aux_x_prime=xp,
                    anchor=d['anchor_xy'][:,0],projection_corrected=d['projection_corrected'][:,0],
                    boundary=d['projected_to_boundary'][:,0],valid=d['box_valid'][:,0],
                    base_corrected=d['base_corrected'][:,0],stationary_atom=d['stationary_atom'][:,0],
                    geometry_kind=kind[:,0],segment_eligible=eligible[:,0],
                    segment_start=d['anchor_xy'][:,0],
                    segment_end=d.get('segment_end', d['anchor_xy'])[:,0],
                    segment_fraction=d.get('segment_fraction', jnp.zeros_like(d['box_valid'], dtype=s.dtype))[:,0],
                    reference_rectangle=d['rectangle'][:,0],box_low=d['box_low'][:,0],box_high=d['box_high'][:,0],
                    response=d['response'],proposal=d['anchor_xy'][:,0]+d['response'],goal=goal)
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
