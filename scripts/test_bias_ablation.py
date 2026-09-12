"""Numerical tests for the bias-free subspace and existing objective semantics."""
import unittest
from types import SimpleNamespace
import jax.numpy as jnp
import numpy as np
from ett.bias_ablation import embed_weights,require_zero_initial,check_subspace
from ett.rollout_return import residual_parameters,antithetic_gradient,descent_step,two_loss_objective


class BiasAblationTests(unittest.TestCase):
  def initial(self):
    return {'diagonal':{'sentinel':jnp.array(7.)},'residual':{'linear':{'w':jnp.zeros((64,2)),'b':jnp.zeros(2)},'torso':{'sentinel':jnp.array(9.)}}}

  def test_exact_subspace_and_zero_bias(self):
    initial=self.initial();weights=jnp.array([.1,-.2,.3,-.4])
    value=residual_parameters(initial,embed_weights(weights));check_subspace(initial,value,weights)
    np.testing.assert_array_equal(value['residual']['linear']['b'],0)
    np.testing.assert_array_equal(value['residual']['linear']['w'][jnp.array([0,32])],weights.reshape(2,2))
    self.assertIs(value['diagonal'],initial['diagonal'])
    bad={**value,'diagonal':{'sentinel':jnp.array(8.)}}
    with self.assertRaises(RuntimeError):check_subspace(initial,bad,weights)

  def test_reject_posthoc_bias_removal_start(self):
    params=self.initial();require_zero_initial(SimpleNamespace(params=params,action_bound=.25))
    params['residual']['linear']['w']=params['residual']['linear']['w'].at[0,0].set(.01)
    with self.assertRaises(ValueError):require_zero_initial(SimpleNamespace(params=params,action_bound=.25))

  def test_four_dimensional_direction_and_descent(self):
    theta=np.array([1.,-2.,.5,-.25]);epsilon=2*np.eye(4);sigma=.2
    positive=((theta+sigma*epsilon)**2).sum(1);negative=((theta-sigma*epsilon)**2).sum(1)
    gradient=antithetic_gradient(positive,negative,epsilon,sigma)
    np.testing.assert_allclose(gradient,2*theta)
    update=descent_step(theta,gradient,.1,.2)
    self.assertLess(np.linalg.norm(update),np.linalg.norm(theta));self.assertLessEqual(np.linalg.norm(update-theta),.200001)
    np.testing.assert_array_equal(embed_weights(update)[:2],0)

  def test_two_term_objective_sign(self):
    first,terms=two_loss_objective(-5.,jnp.array([2.,4.]),1.)
    lower,_=two_loss_objective(-5.,jnp.array([1.,2.]),1.)
    self.assertEqual(len(terms),2);self.assertEqual(float(first),-2.);self.assertLess(float(lower),float(first))

  def test_shared_weight_perturbations(self):
    old=np.random.default_rng(600000).normal(size=(12,4,6)).astype(np.float32)
    new=old[:,:,2:]
    self.assertEqual(new.shape,(12,4,4));np.testing.assert_array_equal(embed_weights(new[0,0])[2:],old[0,0,2:])
    with self.assertRaises(ValueError):embed_weights(jnp.zeros(6))


if __name__=='__main__':unittest.main(verbosity=2)
