"""Boundary and unchanged learner contract checks; fixtures never train actors."""
import unittest
import jax
import jax.numpy as j
import numpy as np
import optax
from scripts.test_policy_improvement import fixture
from ett import pointmaze_full_f4_integration as p
from ett.policy_improvement import replay, mixed_batch
from crl import losses

class IntegrationTests(unittest.TestCase):
    def test_variable_remaining_horizons_and_pairing(self):
        records=[]
        for t in p.CONFIG['root_times']:
            obs,act=fixture(32,50-t)
            records.append(dict(states=obs[:,:,:8],action=act))
        a=p.synthetic_replay(records,4); b=p.synthetic_replay(records,4)
        x,i=a.sample_audited(20000); y,k=b.sample_audited(20000)
        for name in i: np.testing.assert_array_equal(i[name],k[name])
        self.assertTrue(np.all(i['future']<a.lengths[i['episode']]))
        self.assertTrue(np.isfinite(x.observation).all())
        np.testing.assert_array_equal(x.observation[:,8:],a._obs[i['episode'],i['future'],:8])
        for time in [0,1,5]:
            m=(a.lengths[i['episode']]==11)&(i['time']==time)
            expected=.95**np.arange(1,11-time); expected/=expected.sum()
            actual=np.array([np.mean(i['future'][m]==v) for v in range(time+1,11)])
            np.testing.assert_allclose(actual,expected,atol=.09)

    def test_original_nce_logit_bc_objective_on_mixed_fixture(self):
        cfg,net,update=p.learner()
        init,_=losses.build_learner(net,cfg,lambda x:x,optax.adam(cfg.actor_learning_rate,eps=1e-7),optax.adam(cfg.learning_rate,eps=1e-7))
        state=init(jax.random.PRNGKey(13)); obs,act=fixture()
        batch,_=mixed_batch(replay(obs,act,4),replay(obs,act,5),0,np.random.default_rng(6))
        batch=jax.tree.map(j.asarray,batch)
        _,metrics=update(state,batch)
        logits=net.q_network.apply(state.q_params,batch.observation,batch.action)
        expected_nce=j.mean(optax.sigmoid_binary_cross_entropy(logits,j.eye(256)))
        s,g=j.split(batch.observation,[8],axis=1)
        actor_obs=j.concatenate([j.concatenate([s,s]),j.concatenate([g,j.roll(g,1,axis=0)])],axis=1)
        dist=net.policy_network.apply(state.policy_params,actor_obs)
        key=jax.random.split(state.key,4)[3]; a=net.sample(dist,key)
        f=net.q_network.apply(state.q_params,actor_obs,a)
        bc=-j.mean(net.log_prob(dist,j.concatenate([batch.action,batch.action])))
        expected=.5*bc-.5*j.mean(j.diag(f))
        np.testing.assert_allclose(metrics['critic_loss'],expected_nce,rtol=1e-5)
        np.testing.assert_allclose(metrics['bc_nll'],bc,rtol=1e-5)
        np.testing.assert_allclose(metrics['actor_loss'],expected,rtol=1e-5)
        self.assertTrue(np.isfinite(float(metrics['actor_loss'])))

if __name__=='__main__': unittest.main()
