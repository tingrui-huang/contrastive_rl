"""Feature, reward, indexing, split and ranking contracts for return readouts."""
import unittest
import jax
import numpy as np

from crl.networks import make_networks
from ett.return_readout import (return_windows, episode_split, ridge_fit, ridge_predict,
    make_pairs, pair_metrics)
from ett.rollout_return import task_reward, validate_fixed_goal, GOAL


class ReadoutTests(unittest.TestCase):
    def test_representations_reproduce_all_heads_and_scaling(self):
        obs=np.random.default_rng(1).normal(size=(7,16)).astype(np.float32)
        action=np.random.default_rng(2).uniform(-1,1,(7,2)).astype(np.float32)
        for twin,norm,layer in [(False,False,False),(True,True,False),(True,True,True)]:
            network=make_networks(8,8,2,repr_dim=5,hidden_layer_sizes=(8,8),twin_q=twin,
                repr_norm=norm,use_layer_norm=layer,obs_scale=np.linspace(.2,2,16))
            params=network.q_network.init(jax.random.PRNGKey(3))
            if norm: params['~']['repr_log_scale']=np.array(.7,np.float32)
            phi,psi=network.representation_network.apply(params,obs,action)
            value=np.einsum('idh,jdh->ijh',phi,psi)
            if not twin: value=value[...,0]
            original=network.q_network.apply(params,obs,action)
            np.testing.assert_allclose(value,original,rtol=2e-5,atol=2e-5)
            self.assertEqual(phi.shape,(7,5,2 if twin else 1))

    def test_reward_and_fixed_goal_guard(self):
        state=np.tile(np.array([8.5,3.5]*4,np.float32),(4,1))
        state[:,:2]=[[8.5,3.5],[6.5,3.5],[6.5001,3.5],[5.999,3.5]]
        np.testing.assert_array_equal(task_reward(state,np.tile(GOAL,(4,1))),[1,0,1,0])
        with self.assertRaises(ValueError): validate_fixed_goal(np.zeros(8))

    def test_return_indexing_complete_segments_and_dummy_actions(self):
        obs=np.zeros((1,25,16),np.float32);obs[:,:,8:]=GOAL
        obs[0,[1,20,21],:2]=[8.5,3.5]
        actions=np.zeros((1,25,2),np.float32);actions[:,-1]=np.nan
        ep,t,y,_=return_windows(obs,actions,np.array([25]))
        np.testing.assert_array_equal(t,np.arange(5))
        self.assertAlmostEqual(y[0],1+.95**19)
        self.assertAlmostEqual(y[1],.95**18+.95**19)
        self.assertEqual(t[-1]+20,24)
        self.assertEqual(len(return_windows(obs,actions,np.array([20]))[0]),0)

    def test_episode_split_and_original_bank_exclusion(self):
        parts=episode_split(6600,np.arange(256))
        for a in parts:
            for b in parts:
                if a!=b: self.assertEqual(len(np.intersect1d(parts[a],parts[b])),0)
        self.assertFalse(np.isin(parts['test'],np.arange(256)).any())
        self.assertFalse(np.isin(parts['selection'],np.arange(256)).any())

    def test_ridge_train_only_normalization(self):
        x=np.arange(40,dtype=float).reshape(20,2)
        y=x[:,0]*2+3
        val=x+100
        model,detail=ridge_fit(x,y,val,val[:,0]*2+3,[1e-8,1.])
        np.testing.assert_array_equal(model['mean'],x.mean(0))
        self.assertLess(np.max(np.abs(ridge_predict(model,val)-(val[:,0]*2+3))),1e-4)
        self.assertEqual(detail['selected_penalty'],1e-8)

    def test_pairs_disjoint_and_target_ties(self):
        ep=np.repeat(np.arange(8),3);t=np.tile(np.arange(3),8)
        state=np.tile(np.array([3.3,3.5]*4),(24,1));goal=np.tile(GOAL,(24,1))
        pairs=make_pairs(ep,t,state,goal,np.zeros(24),matched=True)
        self.assertEqual(len(pairs),4)
        self.assertEqual(len(np.unique(ep[pairs])),8)
        result=pair_metrics(np.zeros(24),np.arange(24),pairs)
        self.assertIsNone(result['accuracy']);self.assertEqual(result['equal_return_pairs'],4)
        y=np.arange(24,dtype=float)
        self.assertEqual(pair_metrics(y,y,pairs)['accuracy'],1.)
        self.assertEqual(pair_metrics(y,np.zeros(24),pairs)['accuracy'],.5)


if __name__=='__main__': unittest.main()
