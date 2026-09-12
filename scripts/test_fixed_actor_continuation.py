"""Focused native-simulator checks on separate, predeclared check seeds."""
import copy
import unittest
import numpy as np

from ett.fixed_actor_continuation import TimedSimulator, coverage_action, select_contexts
from ett.analyze_fixed_actor_continuation import reliable_pairs, credit


class ContinuationChecks(unittest.TestCase):
    def test_exact_full_snapshot_replay_and_future_rng(self):
        for source in [1,2,3]:
            sim=TimedSimulator(15000000+source)
            rng=np.random.default_rng(15000100+source);memo=dict(speed=.9)
            for t in range(10): sim.step(coverage_action(source,sim.observation(),t,rng,memo))
            saved=sim.snapshot();a=TimedSimulator.restore(saved);b=TimedSimulator.restore(saved)
            for _ in range(20):
                action=rng.uniform(-1,1,2).astype(np.float32)
                aa,ar,at=a.step(action);bb,br,bt=b.step(action)
                np.testing.assert_array_equal(aa,bb);self.assertEqual((ar,at),(br,bt))
                self.assertEqual(a.env._rng.bit_generator.state,b.env._rng.bit_generator.state)
            changed=TimedSimulator.restore(saved,15000200)
            unchanged=TimedSimulator.restore(saved)
            # Audit initial simulator state, never provide these fields to a model.
            for key,value in unchanged.env.__dict__.items():
                if key=='_rng': continue
                other=changed.env.__dict__[key]
                if isinstance(value,(np.ndarray,list)): np.testing.assert_array_equal(value,other)
                else: self.assertEqual(value,other)
            self.assertEqual(changed.elapsed,10)
            np.testing.assert_array_equal(changed.observation(),unchanged.observation())

    def test_complete_horizon_and_absorption_contract(self):
        sim=TimedSimulator(15000010)
        for _ in range(30): sim.step(np.array([1.,0.],np.float32))
        sim.require_horizon(20)
        old=sim.observation().copy();was_absorbing=sim.env._dead
        for i in range(20):
            obs,reward,truncated=sim.step(np.array([1.,0.],np.float32))
            self.assertEqual(truncated,i==19)
            if was_absorbing:
                np.testing.assert_array_equal(obs[:2],old[:2]);self.assertEqual(reward,0)
        self.assertTrue(was_absorbing,'fixed check seed must exercise the absorbing branch')
        with self.assertRaises(ValueError): sim.step(np.zeros(2,np.float32))
        sim=TimedSimulator(15000011)
        for _ in range(31): sim.step(np.zeros(2,np.float32))
        with self.assertRaises(ValueError): sim.require_horizon(20)

    def test_actual_reward_timing_and_f4(self):
        sim=TimedSimulator(15000012);rng=np.random.default_rng(15000013);memo=dict(speed=.9)
        observations=[sim.observation()];rewards=[]
        for t in range(50):
            obs,reward,_=sim.step(coverage_action(2,sim.observation(),t,rng,memo))
            observations.append(obs);rewards.append(reward)
        observations=np.array(observations)
        np.testing.assert_array_equal(observations[1:,2:8],observations[:-1,:6])
        np.testing.assert_array_equal(rewards,(np.linalg.norm(observations[1:,:2]-[8.5,3.5],axis=1)<2).astype(float))
        self.assertGreater(sum(rewards),0,'check seed must exercise nonzero task rewards')
        for t in [0,10,30]:
            target=np.dot(np.array(rewards)[t:t+20],.95**np.arange(20))
            expected=sum(.95**k*float(np.linalg.norm(observations[t+k+1,:2]-[8.5,3.5])<2) for k in range(20))
            self.assertAlmostEqual(target,expected)

    def test_action_alignment_and_bounds(self):
        sim=TimedSimulator(15000014);saved=sim.snapshot();action=np.array([.7,-.1],np.float32)
        left=sim.step(action);right=TimedSimulator.restore(saved).step(action)
        np.testing.assert_array_equal(left[0],right[0]);self.assertEqual(left[1],right[1])
        for invalid in [np.array([1.01,0]),np.array([np.nan,0]),np.zeros(3)]:
            with self.assertRaises(ValueError):sim.step(invalid)

    def test_selection_is_visible_and_fixed(self):
        obs=np.tile(np.array([.5,3.5]*4+[8.5,3.5]*4),(32,1));parent=np.repeat(np.arange(4),8)
        obs[:,0]=np.linspace(.5,8,32)
        a=select_contexts(obs,parent);b=select_contexts(obs.copy(),parent.copy())
        np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(np.bincount(parent[a]),[2]*4)

    def test_uncertain_ordering_and_ties(self):
        returns=np.array([[0]*24,[.1]*24,[2]*24,[0,4]*12],float)
        pairs=np.array([[0,1],[0,2],[2,3]])
        reliable,unequal,_=reliable_pairs(returns,pairs)
        np.testing.assert_array_equal(reliable,[False,True,False])
        np.testing.assert_array_equal(unequal,[True,True,False])
        np.testing.assert_array_equal(credit(np.array([1,-1,1]),np.array([0,-2,-1])),[.5,1,0])


if __name__=='__main__': unittest.main()
