"""Small analytic fixtures; no experiment model samples or actor updates."""
import unittest
import jax
import jax.numpy as j
import numpy as np
from ett import pointmaze_response_search as p

class ResponseTests(unittest.TestCase):
    def test_starts_projection_and_frozen_heads(self):
        base=np.arange(48,dtype=np.float32)/1000
        starts=p.starts(base)
        for theta in starts:
            np.testing.assert_array_equal(theta[:16],base[:16])
            self.assertGreater(np.linalg.norm(theta[16:]),0)
            self.assertLessEqual(np.linalg.norm(p.project_response(theta[16:]*100).reshape(8,2,2),axis=(1,2)).max(),1.000001)
        self.assertFalse(np.array_equal(starts[1],starts[2]))
        theta,m,v,info=p.response_step(base,np.eye(32)[:8],np.tile([1.,-1.],(8,1)),np.zeros(32),np.zeros(32),1)
        np.testing.assert_array_equal(theta[:16],base[:16])
        self.assertLessEqual(info['actual_step_norm'],.250001)

    def test_mc_remaining_horizon_and_first_action(self):
        class Nominal:
            def sample(self,s,key,n,goal):return j.zeros((len(s),2))
        class Engine:
            nominal=Nominal()
            def actor_jit(self,s,key):return j.tile(j.array([-1.,0.]),(len(s),1))
            def _sample(self,theta,s,a,xp,key,n):
                ns=j.concatenate([s[:,:2]+.1*a,s[:,:6]],axis=1)
                return ns[:,None],dict(box_valid=j.ones((len(s),1),bool))
        mc=p.Continuation(Engine());s=np.tile(p.GOAL,(3,1));s[:,0]-=2.05
        first=np.tile([1.,0.],(3,1)).astype(np.float32)
        rewards,valid=mc.run(np.zeros(48,np.float32),s,first,np.array([0,1,2]),jax.random.PRNGKey(5))
        expected=np.zeros((3,48));expected[1:,0]=1
        np.testing.assert_array_equal(rewards,expected)
        self.assertTrue(np.asarray(valid).all())

    def test_paired_uncertainty(self):
        a=np.full((32,8),-.2);roots=np.repeat(np.arange(16),2)
        row=p.uncertainty(a,roots)
        np.testing.assert_allclose(row['root_ci95'],[-.2,-.2])
        np.testing.assert_allclose(row['conditional_mc_ci95'],[-.2,-.2])

if __name__=='__main__':unittest.main()
