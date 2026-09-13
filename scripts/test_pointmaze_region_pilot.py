"""Fixed small component budget; no pilot outcomes used for tuning."""
import unittest
import jax
import numpy as np
import torch
from torch.nn import functional as F

from ett.pointmaze_region_pilot import (
    B, H, GAMMA, CONFIG, GOAL, Kernel, RegionCritic, features, positives,
    task_reward, validate_selected_set, surrogate, positives, fit, parameter_sha)
from ett.eval_pointmaze_region_pilot import paired_interval


class RegionAdapterTests(unittest.TestCase):
    def test_binary_nce_prior_and_remaining_horizon(self):
        p = torch.tensor([.23, .77], dtype=torch.float64)
        f = torch.log(p / ((B-1)*.5)).requires_grad_()
        loss = ((p*F.softplus(-f)).sum() + (B-1)*F.softplus(f).mean())/B
        gradient, = torch.autograd.grad(loss, f)
        np.testing.assert_allclose(gradient, 0., atol=1e-15)
        for h in [0, 1, 5, 10]:
            decoded = (1-GAMMA**h)*(B-1)*.5*np.exp(f.detach().numpy())
            np.testing.assert_allclose(decoded, (1-GAMMA**h)*p, atol=1e-15)
        critic = RegionCritic(123)
        np.testing.assert_array_equal(critic.predict(np.zeros((4,8)), np.zeros((4,2)), np.zeros(4), np.zeros(8), np.ones(8)), 0.)

    def test_positive_timing_and_features(self):
        n = 20000
        record = dict(states=np.zeros((n,H+1,8)), action=np.zeros((n,H,2)), reward=np.zeros((n,H)))
        record['reward'][:, -1] = 1
        x, label = positives(record, 700, np.zeros(8), np.ones(8))
        labels = label.reshape(n,H)
        for t in range(H):
            weights = GAMMA**np.arange(H-t)
            self.assertAlmostEqual(labels[:,t].mean(), weights[-1]/weights.sum(), delta=.013)
        np.testing.assert_array_equal(labels[:,-1], 1)
        np.testing.assert_allclose(x.reshape(n,H,11)[0,:,-1], np.arange(H,0,-1)/H)
        self.assertEqual(CONFIG['root_time']+H, 50)
        self.assertEqual(features(np.zeros((3,8)),np.zeros((3,2)),np.ones(3),np.zeros(8),np.ones(8)).shape, (3,11))

    def test_anisotropic_score_and_cluster_pairing(self):
        # Antithetic coordinate-complete directions give the exact linear gradient.
        scales = np.r_[np.full(16,.01),np.full(32,.1)]
        directions = np.eye(48)*np.sqrt(48)
        known = np.linspace(-2,3,48)
        differences = 2*(directions*scales)@known
        estimate = (differences[:,None]*directions).mean(0)/(2*scales)
        np.testing.assert_allclose(estimate,known,atol=1e-14)
        ci = paired_interval(np.zeros(64),np.arange(64),stratified=True)
        self.assertEqual(ci['ci95'],[0.,0.])


class GeneratorIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = Kernel()
        with np.load('artifacts/ett_convex_adversarial/l1_matrix32_s01_v1/fit_contexts.npz') as z:
            cls.state, cls.action = z['validation_state'][:4], z['validation_action'][:4]

    def test_original_kernel_regression_and_diagonal_invariance(self):
        e, s, a = self.engine, self.state, self.action
        key = jax.random.PRNGKey(761)
        goal = np.broadcast_to(GOAL,s.shape)
        xp = e.nominal.sample(s,key,1,goal=goal)
        original, _ = e.model._sample(np.zeros(32,np.float32),s,a,xp,goal,key,8)
        actual, details = e.sample(np.zeros(48,np.float32),s,a,xp,key,8)
        np.testing.assert_array_equal(original,actual)
        validate_selected_set(s,actual,details)
        theta = np.random.default_rng(715).normal(0,.03,48).astype(np.float32)
        y, _ = e.sample(theta,s,a,a,key,8)
        theta[16:] = 40
        changed, _ = e.sample(theta,s,a,a,key,8)
        np.testing.assert_array_equal(y,changed)

    def test_full_action_coupling_and_changed_diagonal(self):
        e, s = self.engine,self.state
        key = jax.random.PRNGKey(771)
        xp = np.zeros((4,2),np.float32)
        theta = np.random.default_rng(716).normal(0,1,48).astype(np.float32)
        actions = [np.full((4,2),-1,np.float32),np.full((4,2),1,np.float32),np.full((4,2),.999,np.float32)]
        outputs = []
        for action in actions:
            y,d = e.sample(theta,s,action,xp,key,8)
            validate_selected_set(s,y,d); outputs.append(np.asarray(y))
        for i in range(3):
            for k in range(i):
                self.assertLessEqual(float((np.linalg.norm(outputs[i]-outputs[k],axis=-1)-np.linalg.norm(actions[i]-actions[k],axis=-1)[:,None]).max()),2e-6)

    def test_rollout_replay_action_reward_and_f4(self):
        e,s = self.engine,self.state
        first = np.array([[-1.,1.],[1.,-1.],[0.,0.],[.3,-.4]],np.float32)
        key = jax.random.PRNGKey(781)
        r = e.rollout(np.zeros(48,np.float32),s,key,H,first)
        replay = e.rollout(np.zeros(48,np.float32),s,key,H,first)
        for k in r: np.testing.assert_array_equal(r[k],replay[k])
        np.testing.assert_array_equal(r['action'][:,0],first)
        self.assertFalse(np.array_equal(r['x_prime'][:,0],first))
        np.testing.assert_array_equal(r['states'][:,1:,2:],r['states'][:,:-1,:6])
        expected = np.asarray(task_reward(r['states'][:,1:],np.broadcast_to(GOAL,r['states'][:,1:].shape)))
        np.testing.assert_array_equal(r['reward'],expected)
        np.testing.assert_array_equal(expected,(np.linalg.norm(r['states'][:,1:,:2]-GOAL[:2],axis=-1)<2).astype(np.float32))
        self.assertEqual(r['states'].shape,(4,11,8))

    def test_frozen_surrogate_smoke(self):
        torch.set_num_threads(1)
        e,s = self.engine,self.state
        theta = np.zeros(48,np.float32)
        record = e.rollout(theta,s,jax.random.PRNGKey(791))
        critic = RegionCritic(797)
        opt = torch.optim.Adam(critic.parameters(),lr=.003)
        mean,std = np.zeros(8),np.ones(8)
        loss = fit(critic,opt,positives(record,798,mean,std),3,799)
        self.assertTrue(np.isfinite(loss))
        before = parameter_sha(critic)
        class Counter:
            def __init__(self): self.total = 0
            def add(self,n,purpose): self.total += n
        ledger = Counter()
        a = surrogate(e,theta,critic,record,mean,std,801,ledger)
        b = surrogate(e,theta,critic,record,mean,std,801,ledger)
        self.assertEqual(a,b)
        self.assertEqual(parameter_sha(critic),before)
        self.assertFalse(any(p.requires_grad for p in critic.parameters()))
        self.assertEqual(ledger.total,1024)


if __name__ == '__main__':
    unittest.main()
