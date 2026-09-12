"""Hard task-return rollout and low-dimensional residual search for PointMaze.

This module is specific to the fixed goal in windy F4 v0. It never reads audit
labels or a critic. Parameter perturbations include every discrete sampling
and geometry decision in the resulting trajectory return.
"""
import jax
import jax.numpy as jnp
import numpy as np

ENVIRONMENT = 'point_two_route_swamp_windy_f4_v0'
GOAL = np.tile(np.array([8.5,3.5],np.float32),4)
START = np.tile(np.array([.5,3.5],np.float32),4)
HORIZON = 50
DISCOUNT = .95
HEAD_ROWS = (0,32)
SEARCH_DIMENSION = 6


def validate_fixed_goal(goal):
  goal = np.asarray(goal)
  if goal.shape[-1] != 8 or not np.all(goal == GOAL):
    raise ValueError('visible reward is justified only for this fixed PointMaze goal')


def task_reward(next_state, commanded_goal):
  """Hard r(s_next)=1[||newest XY-goal XY||<2] for the verified fixed task.

  Actual absorbing deaths occur only in cells x in [3,6), y in [3,4), at
  least 2.5 units from fixed goal (8.5,3.5). Thus the environment's hidden
  death gate cannot alter this position reward on reachable real states.
  Synthetic observations need not be physically realizable. Do not generalize
  this function to another goal/task; public rollout entrypoints validate it.
  """
  return (jnp.linalg.norm(next_state[...,:2]-commanded_goal[...,:2],axis=-1)<2.).astype(jnp.float32)


def discounted_return(rewards, discount=DISCOUNT):
  return jnp.sum(rewards*jnp.power(discount,jnp.arange(rewards.shape[-1])),axis=-1)


def two_loss_objective(diagonal_nll, returns, weight=1.):
  """Minimize likelihood loss plus POSITIVE expected return: exactly two terms."""
  off = jnp.mean(returns)
  return diagonal_nll+weight*off, (diagonal_nll,off)


def residual_parameters(initial, theta):
  """Six coordinates: final-layer b[2], and w rows 0,32 [2,2].

  All other residual weights and the entire diagonal model are frozen. The
  original residual MLP, tanh, action radius, projection and fallback remain.
  theta=0 exactly reproduces the diagonal-only initialized checkpoint.
  """
  if theta.shape != (SEARCH_DIMENSION,):
    raise ValueError('expected six residual-head search coordinates')
  residual = dict(initial['residual'])
  head = dict(residual['linear'])
  head['b'] = jnp.asarray(head['b'])+theta[:2]
  head['w'] = jnp.asarray(head['w']).at[jnp.array(HEAD_ROWS)].add(theta[2:].reshape(2,2))
  residual['linear'] = head
  return {'diagonal':initial['diagonal'],'residual':residual}


def antithetic_gradient(positive, negative, perturbations, sigma, weight=1.):
  """Gaussian-smoothed objective gradient, paired +/- and common randomness.

  Each scalar is a complete mean rollout return. This does not differentiate
  a hard reward, categorical draw or geometric branch. It estimates the
  smoothed expected-return derivative including their distributional effects.
  """
  if sigma <= 0:
    raise ValueError('sigma must be positive')
  positive, negative, perturbations = map(np.asarray,(positive,negative,perturbations))
  if positive.ndim!=1 or positive.shape!=negative.shape or perturbations.shape[0]!=len(positive):
    raise ValueError('one positive and negative scalar per perturbation required')
  return weight*np.mean(((positive-negative)/(2*sigma))[:,None]*perturbations,axis=0)


def descent_step(theta, gradient, learning_rate=.1, max_step_norm=.2):
  update = learning_rate*np.asarray(gradient)
  update *= min(1.,max_step_norm/max(np.linalg.norm(update),1e-12))
  return (np.asarray(theta)-update).astype(np.float32)


class TaskRollout:
  """Independent model trajectories [context,replicate,time,...], fixed goal.

  A remaining-horizon mask supports observed continuation diagnostics. The
  primary experiment always starts from reset and executes all 50 transitions.
  actions stores only the agent's x. x_prime is returned in separate auxiliary
  diagnostics, never as an agent execution action.
  """
  def __init__(self,model,nominal,actor,horizon=HORIZON,discount=DISCOUNT):
    self.model,self.nominal,self.actor=model,nominal,actor
    self.horizon,self.discount=int(horizon),float(discount)
    if not 1<=self.horizon<=50:
      raise ValueError('horizon must be in [1,50]')
    initial = model.params
    def generate(theta,states,goals,remaining,key):
      params = residual_parameters(initial,theta)
      def step(state,arguments):
        t,k = arguments
        nominal_key,actor_key,transition_key=jax.random.split(k,3)
        xp=nominal.sample(state,nominal_key,1,goal=goals)
        action=actor(state,goals,actor_key)
        sampled,detail=model.sample_flat(params,state,action,xp,goals,transition_key,1)
        active=t<remaining
        next_state=jnp.where(active[:,None],sampled[:,0],state)
        reward=jnp.where(active,task_reward(next_state,goals),0.)
        record={'next_state':next_state,'action':jnp.where(active[:,None],action,0.),
            'reward':reward,'active':active,
            'aux_observational_action':jnp.where(active[:,None],xp,0.),
            **{name:value[:,0] for name,value in detail.items()}}
        return next_state,record
      _,records=jax.lax.scan(step,states,(jnp.arange(self.horizon),jax.random.split(key,self.horizon)))
      records={name:jnp.swapaxes(value,0,1) for name,value in records.items()}
      records['states']=jnp.concatenate([states[:,None],records.pop('next_state')],axis=1)
      records['return']=discounted_return(records['reward'],self.discount)
      records['commanded_goal']=goals
      return records
    self._generate=jax.jit(generate)

  def run(self,theta,states,goals,key,replicates,remaining=None):
    validate_fixed_goal(goals)
    states,goals=np.asarray(states,np.float32),np.asarray(goals,np.float32)
    if states.ndim!=2 or states.shape[-1]!=8 or goals.shape!=states.shape or replicates<1:
      raise ValueError('expected matching [B,8] state/goal and positive replicate count')
    if remaining is None:remaining=np.full(len(states),self.horizon,np.int32)
    remaining=np.asarray(remaining,np.int32)
    if remaining.shape!=(len(states),) or np.any((remaining<0)|(remaining>50)):
      raise ValueError('invalid remaining episode horizon')
    repeat=lambda v:np.repeat(v,replicates,axis=0)
    values=self._generate(jnp.asarray(theta),repeat(states),repeat(goals),repeat(remaining),key)
    return {name:np.asarray(value).reshape((len(states),replicates)+value.shape[1:]) for name,value in values.items()}
