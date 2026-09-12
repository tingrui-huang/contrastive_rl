"""Numerical tests independent of trained PointMaze checkpoint availability."""
import unittest
import jax
import jax.numpy as jnp
import numpy as np

from crl.envs import TwoRouteSwampWindyF4Env
from ett.rollout_return import (GOAL,START,TaskRollout,task_reward,validate_fixed_goal,
    discounted_return,two_loss_objective,antithetic_gradient,descent_step,residual_parameters)


class NominalStub:
  def sample(self,state,key,count,goal):
    return jnp.stack([jnp.ones(len(state))*.15,jnp.zeros(len(state))],axis=-1)


class TransitionStub:
  params={'diagonal':{'sentinel':jnp.array(3.)},'residual':{'linear':{'b':jnp.zeros(2),'w':jnp.zeros((64,2))}}}
  def sample_flat(self,params,state,action,xp,goal,key,count):
    # Distinct action arguments make accidental substitution visible.
    next_xy=state[:,:2]+action+2*xp
    return jnp.concatenate([next_xy,state[:,:6]],-1)[:,None],{'anchor_xy':next_xy[:,None],'radius':jnp.zeros((len(state),1))}


class RolloutReturnTests(unittest.TestCase):
  def test_hard_reward_and_specific_goal_contract(self):
    states=jnp.array([[6.5,3.5]*4,[6.5001,3.5]*4,[5.999,3.5]*4,[8.5,3.5]*4])
    np.testing.assert_array_equal(task_reward(states,GOAL),[0,1,0,1])
    with self.assertRaises(ValueError):validate_fixed_goal(np.zeros((1,8)))
    # Actual simulator trajectories: reported reward agrees without reading U/death.
    for seed in range(6):
      env=TwoRouteSwampWindyF4Env(seed=seed)
      obs=env.reset()
      for step in range(50):
        action=np.array([1.,0.],np.float32) if step<10 else np.zeros(2,np.float32)
        obs,reward,done,_=env.step(action)
        self.assertEqual(reward,float(task_reward(obs[:8],obs[8:])))
        self.assertFalse(done)

  def test_return_sign_and_diagonal_constant(self):
    rewards=jnp.array([[1.,0.,1.],[0.,1.,0.]])
    np.testing.assert_allclose(discounted_return(rewards,.5),[1.25,.5])
    loss,terms=two_loss_objective(-3.,discounted_return(rewards,.5),2.)
    self.assertAlmostEqual(float(loss),-1.25)
    self.assertEqual(len(terms),2)
    smaller,_=two_loss_objective(-3.,jnp.zeros(2),2.)
    self.assertLess(float(smaller),float(loss))

  def test_paired_es_direction_and_descent(self):
    theta=np.array([1.,-2.]);eps=np.eye(2)*np.sqrt(2);sigma=.2
    positive=np.square(theta+sigma*eps).sum(1)
    negative=np.square(theta-sigma*eps).sum(1)
    gradient=antithetic_gradient(positive,negative,eps,sigma)
    np.testing.assert_allclose(gradient,2*theta)
    updated=descent_step(theta,gradient)
    self.assertLess(np.square(updated).sum(),np.square(theta).sum())
    np.testing.assert_array_equal(antithetic_gradient(positive,positive,eps,sigma),0.)

  def test_residual_subspace_preserves_backbone(self):
    initial=TransitionStub.params
    modified=residual_parameters(initial,jnp.arange(6.))
    self.assertIs(modified['diagonal'],initial['diagonal'])
    np.testing.assert_array_equal(modified['residual']['linear']['b'],[0,1])
    np.testing.assert_array_equal(modified['residual']['linear']['w'][jnp.array([0,32])],[[2,3],[4,5]])
    np.testing.assert_array_equal(initial['residual']['linear']['w'],0.)

  def test_grouping_fixed_goal_history_and_action_roles(self):
    actor=lambda state,goal,key:jnp.stack([jnp.ones(len(state))*.2,jnp.zeros(len(state))],-1)
    rollout=TaskRollout(TransitionStub(),NominalStub(),actor,horizon=3)
    states=np.stack([START,np.tile(np.array([6.4,3.5],np.float32),4)])
    result=rollout.run(np.zeros(6,np.float32),states,np.broadcast_to(GOAL,states.shape),jax.random.PRNGKey(7),4,remaining=[3,1])
    self.assertEqual(result['states'].shape,(2,4,4,8))
    self.assertEqual(result['return'].shape,(2,4))
    np.testing.assert_allclose(result['action'][0,...,0],.2)
    np.testing.assert_allclose(result['aux_observational_action'][0,...,0],.15)
    np.testing.assert_allclose(result['states'][0,:,1,0],1.)
    before,after=result['states'][:,:,:-1],result['states'][:,:,1:]
    np.testing.assert_array_equal(after[...,2:][result['active']],before[...,:6][result['active']])
    np.testing.assert_array_equal(result['reward'][1,:,1:],0.)
    np.testing.assert_array_equal(result['commanded_goal'],np.broadcast_to(GOAL,(2,4,8)))
    repeated=rollout.run(np.zeros(6,np.float32),states,np.broadcast_to(GOAL,states.shape),jax.random.PRNGKey(7),4,remaining=[3,1])
    for k in result:np.testing.assert_array_equal(result[k],repeated[k])


if __name__=='__main__':unittest.main(verbosity=2)
