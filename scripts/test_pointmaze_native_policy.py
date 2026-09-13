"""Gradient correctness, executed-action diagonal routing and trust guards."""
import unittest
from types import SimpleNamespace

import jax
import jax.numpy as j
import numpy as np
import torch

from crl import networks
from ett import pointmaze_native_policy as p
from ett import eval_pointmaze_native_policy as e
from ett import pointmaze_early_pilot as early
from ett import pointmaze_region_pilot as old


class NativePolicyTests(unittest.TestCase):
    def test_jax_decode_and_action_gradient_match_torch(self):
        torch.set_num_threads(1); critic=early.Critic(19); rng=np.random.default_rng(21)
        state=rng.normal(size=(4,8)).astype(np.float32); action=rng.uniform(-.8,.8,(4,2)).astype(np.float32)
        h=np.array([0,1,49,50]); mean=np.zeros(8,np.float32); std=np.ones(8,np.float32)
        weights=p.export_critic(critic)
        actual=p.decoded_q(weights,j.asarray(state),j.asarray(action),j.asarray(h),j.asarray(mean),j.asarray(std))
        target=critic.predict(state,action,h,mean,std)
        np.testing.assert_allclose(actual,target,rtol=2e-6,atol=2e-6)
        a=torch.tensor(action,dtype=torch.float64,requires_grad=True)
        x=torch.cat([torch.tensor(state,dtype=torch.float64),a,torch.tensor(h/50.,dtype=torch.float64)[:,None]],dim=1)
        q=torch.tensor((1-.95**h)*15.5)*torch.exp(critic(x)[:,1])
        grad=torch.autograd.grad(q.sum(),a)[0].detach().numpy()
        jgrad=jax.grad(lambda a:j.sum(p.decoded_q(weights,j.asarray(state),a,j.asarray(h),j.asarray(mean),j.asarray(std))))(j.asarray(action))
        np.testing.assert_allclose(jgrad,grad,rtol=3e-5,atol=2e-6)
        np.testing.assert_array_equal(jgrad[0],0); self.assertGreater(np.linalg.norm(jgrad[1:]),1e-5)

    def test_reparameterized_actor_gradient_matches_central_difference(self):
        # Double precision only inside this numerical test; experiment stays float32.
        with jax.enable_x64(True):
            net=networks.make_networks(8,8,2)
            params=jax.tree.map(lambda x:j.asarray(x,dtype=j.float64),net.policy_network.init(jax.random.PRNGKey(4)))
            critic=jax.tree.map(lambda x:j.asarray(x,dtype=j.float64),p.export_critic(early.Critic(7)))
            rng=np.random.default_rng(9); state=j.asarray(rng.normal(size=(8,8)))
            h=j.asarray([1,5,10,20,30,40,49,50]); weight=j.ones(8)
            half=rng.normal(size=(8,2,2)); eps=j.asarray(np.concatenate([half,-half],axis=1))
            fn=lambda params:p.actor_loss(params,critic,state,h,weight,eps,j.zeros(8),j.ones(8),net)
            gradient=jax.grad(fn)(params)
            norm=float(j.sqrt(sum(j.sum(v*v) for v in jax.tree.leaves(gradient))))
            self.assertGreater(norm,1e-5)
            direction=jax.tree.map(lambda g:g/norm,gradient)
            plus=jax.tree.map(lambda a,b:a+1e-5*b,params,direction)
            minus=jax.tree.map(lambda a,b:a-1e-5*b,params,direction)
            finite=float((fn(plus)-fn(minus))/(2e-5))
            self.assertAlmostEqual(finite/norm,1.,places=4)

    def test_B_routes_both_channels_to_execution_and_actor_is_explicit(self):
        class Nominal:
            def sample(self,s,key,count,goal): return j.full((len(s),2),-.5)
        class Engine:
            nominal=Nominal()
            def _sample(self,theta,s,a,xp,key,count):
                xy=s[:,:2]+.1*xp
                return j.concatenate([xy,s[:,:6]],axis=1)[:,None],dict(box_valid=j.ones((len(s),1),bool))
        def apply(params,obs):
            return networks.TanhNormalParams(j.broadcast_to(params,(len(obs),2)),j.full((len(obs),2),1e-6))
        net=SimpleNamespace(policy_network=SimpleNamespace(apply=apply)); generator=p.FrozenRollouts(Engine(),net)
        state=np.tile(p.START,(3,1)); theta=np.zeros(48,np.float32); key=jax.random.PRNGKey(6)
        a=generator.rollout(j.array([.5,.1]),theta,True,state,key,3)
        b=generator.rollout(j.array([-.5,.1]),theta,True,state,key,3)
        c=generator.rollout(j.array([.5,.1]),theta,False,state,key,3)
        np.testing.assert_array_equal(a['x_prime'],a['action'])
        np.testing.assert_array_equal(c['x_prime'],-.5)
        self.assertGreater(np.max(np.abs(a['states']-b['states'])),.05)
        self.assertGreater(np.max(np.abs(a['states']-c['states'])),.05)

    def test_initial_actor_reproduces_frozen_original(self):
        net,params=p.actor_setup(); engine=old.Kernel(); states=np.tile(p.START,(16,1))
        key=jax.random.PRNGKey(32)
        actual=jax.jit(lambda s:p.sample_actor(net,params,s,key))(states)
        np.testing.assert_array_equal(actual,engine.actor_jit(states,key))

    def test_exact_KL_and_rejection_rule(self):
        a=networks.TanhNormalParams(j.zeros((2,2)),j.ones((2,2)))
        b=networks.TanhNormalParams(j.full((2,2),.1),j.ones((2,2)))
        np.testing.assert_allclose(p.gaussian_kl(a,b),.01,atol=1e-7)
        self.assertTrue(p.policy_admissible(np.zeros(4),np.zeros(4),0))
        for value in [[.021,0,0,0],[0,.051,0,0],[0,0,.151,0],[np.nan,0,0,0]]:
            self.assertFalse(p.policy_admissible(value,np.zeros(4),0))
        self.assertFalse(p.policy_admissible(np.zeros(4),np.zeros(4),.0021))

    def test_binary_discordance_and_ties_are_not_equivalence(self):
        a=dict(returns=np.array([1.,2.,3.,4.]),success=np.array([1.,0,1,0]),failure=np.array([0.,1,0,1]))
        b=dict(returns=np.array([0.,2.,3.,4.]),success=np.array([0.,0,1,1]),failure=np.array([1.,1,0,0]))
        result=e.paired(a,b)
        self.assertEqual(result['success']['a_one_b_zero'],1);self.assertEqual(result['success']['a_zero_b_one'],1)
        ties=e.paired(a,a)
        self.assertTrue(ties['returns']['observed_all_ties'])
        np.testing.assert_array_equal(ties['returns']['ci95'],[0.,0.])


if __name__=='__main__':unittest.main()
