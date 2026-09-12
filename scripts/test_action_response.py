"""Numerical action-probe checks independent of trained checkpoint availability."""
import copy
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from crl.envs import TwoRouteSwampWindyF4Env
from ett.action_reference import snapshot,restore,StepNoise,continuation_indices,prescribed_reference
from ett.action_response import actions_for,InstrumentedProbe,response_metrics,select_contexts,wall_distance
from ett.anchored_transition import AnchoredTransition
from ett.diagonal_transition import DiagonalTransitionModel,DiagonalTransitionSpec,make_transition_network
from ett.rollout_return import GOAL


class ActionResponseTests(unittest.TestCase):
  def test_continuation_respects_original_horizon(self):
    rows=[{'timestep':t} for t in (0,45,46,48,49)]
    np.testing.assert_array_equal(continuation_indices(rows,5),[0,1])
    np.testing.assert_array_equal(continuation_indices(rows,1),np.arange(5))
    with self.assertRaises(ValueError):prescribed_reference(rows,np.zeros((5,1,2)),horizon=5)

  def test_action_grid_and_no_silent_clipping(self):
    xp=np.array([[1.,-.3],[0,0]],np.float32)
    actions=actions_for(xp)
    self.assertEqual(actions.shape,(2,10,2))
    np.testing.assert_array_equal(actions[:,-1],xp)
    self.assertTrue((np.abs(actions)<=1).all())
    with self.assertRaises(ValueError):actions_for(np.array([[1.01,0]]))

  def test_context_rules_do_not_identify_death(self):
    state=np.tile([[7.,3.5],[2.5,3.98],[2.5,3.5],[4.,3.5]],(1,4)).astype(np.float32)
    state[:3,2]-=.1
    ids,groups,_=select_contexts(state,np.arange(4),np.ones(4),1,0)
    self.assertEqual(list(groups),['open_corridor','boundary','before_swamp','stationary_nonreset'])
    self.assertEqual(ids[-1],3)
    np.testing.assert_allclose(wall_distance(np.array([[3.5,3.5],[3.5,3.98]])),[.5,.02])

  def test_full_simulator_restore(self):
    env=TwoRouteSwampWindyF4Env(seed=44)
    env.step(np.array([.5,.1]));saved=snapshot(env)
    for action in ([0,0],[1,0],[-1,.5]):
      first,second=restore(saved),restore(copy.deepcopy(saved))
      for _ in range(4):
        a=first.step(action);b=second.step(action)
        np.testing.assert_array_equal(a[0],b[0]);self.assertEqual(a[1:3],b[1:3])
        self.assertEqual(first._rng.bit_generator.state,second._rng.bit_generator.state)
    np.testing.assert_array_equal(env.state,saved['state'])

  def test_time_aligned_exogenous_slots(self):
    live=TwoRouteSwampWindyF4Env(seed=22);absorbing=restore(snapshot(live));absorbing._dead=True
    uniforms=np.array([.1,.5,.2]);noise=np.array([.3,-.8])
    for env in (live,absorbing):env._rng=StepNoise(noise,uniforms)
    live.step([.1,0]);absorbing.step([.1,0])
    np.testing.assert_array_equal(live._bits,absorbing._bits)
    np.testing.assert_array_equal(live._bits,[True,False,True])

  def test_instrumentation_invariance_and_common_drift_metric(self):
    spec=DiagonalTransitionSpec(hidden_sizes=(8,8),num_components=2)
    params=make_transition_network(spec).init(jax.random.PRNGKey(1),jnp.zeros((1,20)))
    diagonal=DiagonalTransitionModel(params,spec,np.zeros(20),np.ones(20),np.zeros(2),np.ones(2)*.1)
    model=AnchoredTransition.initialize(diagonal,jax.random.PRNGKey(2),.25)
    state=np.tile([[2.5,3.5],[7.5,3.98]],(1,4)).astype(np.float32)
    goal=np.broadcast_to(GOAL,state.shape);xp=np.array([[.5,0],[-.5,.1]],np.float32)
    record,checks=InstrumentedProbe(model).run(state,xp,goal,jax.random.PRNGKey(3),8)
    self.assertTrue(checks['unmodified_sampler_fixed_key_equality'])
    self.assertLessEqual(response_metrics(record)['max_fixed_key_execution_action_effect'],2e-6)
    model.params['residual']['linear']['b']=jnp.array([2.,-1.])
    record,_=InstrumentedProbe(model).run(state,xp,goal,jax.random.PRNGKey(3),8)
    self.assertAlmostEqual(response_metrics(record)['common_direction_energy_fraction'],1.,places=6)


if __name__=='__main__':unittest.main(verbosity=2)
