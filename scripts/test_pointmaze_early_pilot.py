"""Focused early-context checks without inspecting pilot selection/outcomes."""
import unittest
import jax
import numpy as np
import torch
from ett import pointmaze_early_pilot as p
from ett import eval_pointmaze_early_pilot as e
from ett import pointmaze_region_pilot as old


class EarlyContextTests(unittest.TestCase):
    def test_observable_selection_future_independence_episode_separation(self):
        obs=np.zeros((80,51,16),np.float32);obs[:,:,8:]=old.GOAL
        # Synthetic fixture only: exercise selector access, not model contexts.
        for i in range(80):
            xy=np.array([[2.,3.5],[4.,3.5],[2.,2.]])[i%3]
            obs[i,:,:8]=np.tile(xy,4)
            obs[i,:,2]=xy[0]-.2
        a,c=p.select_roots(obs,np.arange(40),321)
        b,d=p.select_roots(obs,np.arange(40,80),322)
        self.assertEqual(len(a),36);self.assertEqual(len(b),36)
        self.assertEqual(len(np.unique(a[:,0])),36)
        self.assertFalse(np.intersect1d(a[:,0],b[:,0]).size)
        changed=obs.copy();changed[:,11:,:8]=1234
        np.testing.assert_array_equal(a,p.select_roots(changed,np.arange(40),321)[0])
        # A visible previous goal visit disqualifies an otherwise eligible root.
        changed=obs.copy();changed[0,0,:8]=old.GOAL
        self.assertFalse(np.any(p.select_roots(changed,np.arange(40),321)[0][:,0]==0))

    def test_horizon_decode_and_mc_positive_timing(self):
        torch.set_num_threads(1)
        critic=p.Critic(100)
        h=np.array([0,1,40,44,47,49]);s=np.zeros((6,8));a=np.zeros((6,2))
        x=p.features(s,a,h,np.zeros(8),np.ones(8))
        np.testing.assert_array_equal(x[:,-1],h/50)
        with torch.no_grad():logits=critic(torch.from_numpy(x)).numpy()[:,1]
        expected=(1-.95**h)*31*.5*np.exp(logits)
        np.testing.assert_allclose(critic.predict(s,a,h,np.zeros(8),np.ones(8)),expected)
        records=[]
        for length in [40,49]:
            n=4000;r=dict(states=np.zeros((n,length+1,8)),action=np.zeros((n,length,2)),reward=np.zeros((n,length)))
            r['reward'][:,-1]=1;records.append(r)
        x,y=p.positives(records,417,np.zeros(8),np.ones(8));offset=0
        for r in records:
            length=r['action'].shape[1];labels=y[offset:offset+4000*length].reshape(4000,length)
            np.testing.assert_array_equal(labels[:,-1],1)
            w=.95**np.arange(length)
            self.assertAlmostEqual(labels[:,0].mean(),w[-1]/w.sum(),delta=.01)
            self.assertEqual(x[offset,-1],length/50);offset+=4000*length

    def test_recovery_is_not_zero_return_or_death(self):
        # One true delayed recovery, one zero-return stationary path, one
        # monotonically progressing goal path: labels must stay distinct.
        h=8;states=np.zeros((3,h+1,8),np.float32)
        states[0,:,:2]=[[4.,3.5],[3.5,3.5],[4.,3.5],[4.5,3.5],[5.,3.5],[6.,3.5],[7.,3.5],[7.5,3.5],[8.,3.5]]
        states[1,:,:2]=[4.,3.5]
        states[2,:,:2]=np.c_[np.linspace(4.,8.,h+1),np.full(h+1,3.5)]
        reward=np.asarray(old.task_reward(states[:,1:],np.broadcast_to(old.GOAL,states[:,1:].shape)))
        r=dict(states=states,action=np.zeros((3,h,2)),x_prime=np.zeros((3,h,2)),reward=reward,root=np.arange(3))
        arrays=e.path_statistics([r],3,1)
        np.testing.assert_array_equal(arrays['recovered'][:,0],[True,False,False])
        self.assertEqual(arrays['returns'][1,0],0.)
        self.assertNotIn('death',arrays)
        self.assertTrue(np.isnan(arrays['states'][:,:,h+1:]).all())
        np.testing.assert_allclose(arrays['returns'][:,0],.05*reward@.95**np.arange(h))

    def test_variable_horizon_visitation_weight(self):
        # Uniform-root / uniform-time expectation equals discounted sum,
        # whereas uniform transitions would overweight longer root horizons.
        lengths=[40,49];values=[np.linspace(0,1,h) for h in lengths]
        weighted=np.mean([np.mean(h*.95**np.arange(h)*v) for h,v in zip(lengths,values)])
        exact=np.mean([np.sum(.95**np.arange(h)*v) for h,v in zip(lengths,values)])
        self.assertAlmostEqual(weighted,exact)
        ci=e.interval(np.zeros(36),np.arange(36),np.repeat(np.arange(3),12))
        self.assertEqual(ci['ci95'],[0.,0.])

    def test_collection_horizon_action_alignment_and_frozen_surrogate(self):
        torch.set_num_threads(1);engine=old.Kernel()
        with np.load('artifacts/ett_convex_adversarial/l1_matrix32_s01_v1/fit_contexts.npz') as z:s=z['validation_state'][:2]
        class Counter:
            def __init__(self):self.total=0
            def add(self,n,purpose):self.total+=n
        ledger=Counter();first=np.array([[.25,-.5],[-.5,.25]],np.float32)
        records=p.collect(engine,np.zeros(48,np.float32),s,np.array([1,10]),440,ledger,'test',2,first)
        self.assertEqual(ledger.total,2*(49+40))
        for r in records:
            np.testing.assert_array_equal(r['action'][:,0],first[r['root']])
            np.testing.assert_array_equal(r['states'][:,1:,2:],r['states'][:,:-1,:6])
        critic=p.Critic(446);opt=torch.optim.Adam(critic.parameters(),lr=.003)
        old.fit(critic,opt,p.positives(records,447,np.zeros(8),np.ones(8)),3,448)
        before=old.parameter_sha(critic)
        a=p.surrogate(engine,np.zeros(48,np.float32),critic,records,np.zeros(8),np.ones(8),449,ledger)
        b=p.surrogate(engine,np.zeros(48,np.float32),critic,records,np.zeros(8),np.ones(8),449,ledger)
        self.assertEqual(a,b);self.assertEqual(old.parameter_sha(critic),before)
        self.assertEqual(ledger.total,178+1024)


if __name__=='__main__':unittest.main()
