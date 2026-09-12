"""Analytic sign, F4, diagonal identity and full-action constraint checks."""
import unittest
import jax.numpy as jnp
import numpy as np

from ett.convex_action_transition import emit,matrices,energy_score,objective,convex_box
from ett.diagonal_transition import POINTMAZE_WALLS
from ett.rollout_return import antithetic_gradient,descent_step


class ConvexChecks(unittest.TestCase):
    def test_analytic_descent_sign_and_zero_weight(self):
        d=32;theta=np.linspace(-.4,.3,d);target=np.linspace(.5,-.2,d);sigma=.1
        directions=np.eye(d)*np.sqrt(d)
        loss=lambda z:np.sum((z-target)**2,axis=-1)
        plus=loss(theta+sigma*directions);minus=loss(theta-sigma*directions)
        grad=antithetic_gradient(plus,minus,directions,sigma,1.)
        np.testing.assert_allclose(grad,2*(theta-target),atol=1e-12)
        updated=descent_step(theta,grad,learning_rate=.1,max_step_norm=1.)
        self.assertLess(loss(updated),loss(theta))
        np.testing.assert_array_equal(antithetic_gradient(plus,minus,directions,sigma,0),np.zeros(d))
        self.assertAlmostEqual(float(objective(.3,jnp.array([2.,4.]),.1)),.6,places=6)

    def test_matrices_and_projection_all_pairs(self):
        rng=np.random.default_rng(19000000)
        cells=np.argwhere(POINTMAZE_WALLS==0);xy=cells+.5
        s=np.tile(xy,(1,4)).astype(np.float32);anchor=s[:,None,:2]
        xp=rng.uniform(-1,1,(len(s),2)).astype(np.float32)
        theta=rng.normal(size=32)*8
        self.assertLessEqual(np.linalg.svd(np.asarray(matrices(theta)),compute_uv=False).max(),1+1e-6)
        actions=rng.uniform(-1,1,(40,len(s),2)).astype(np.float32)
        outputs=[]
        for a in actions:
            y,d=emit(theta,s,a,xp,anchor);y=np.asarray(y);outputs.append(y[:,0,:2])
            self.assertTrue(np.asarray(d['box_valid']).all())
            np.testing.assert_array_equal(y[:,0,2:],s[:,:6])
            self.assertTrue(np.all(np.abs(y[:,0,:2]-s[:,:2])<=1+1e-6))
            cell=np.floor(y[:,0,:2]).astype(int)
            self.assertTrue(np.all(POINTMAZE_WALLS[cell[:,0],cell[:,1]]==0))
        outputs=np.array(outputs)
        od=np.linalg.norm(outputs[:,None]-outputs[None,:],axis=-1)
        ad=np.linalg.norm(actions[:,None]-actions[None,:],axis=-1)
        self.assertLessEqual(np.max(od-ad),2e-6)

    def test_diagonal_identity_and_near_wall_continuity(self):
        s=np.tile([2.9999,3.0001],4).astype(np.float32)[None]
        anchor=np.array([[[3.0001,3.0001],[2.9999,3.0001]]],np.float32)
        xp=np.array([[.2,-.2]],np.float32);theta=np.arange(32,dtype=np.float32)-10
        y,_=emit(theta,s,xp,xp,anchor)
        np.testing.assert_array_equal(y[...,:2],anchor)
        for gap in [1e-3,1e-5]:
            y1,_=emit(theta,s,xp+gap,xp,anchor)
            self.assertLessEqual(np.linalg.norm(np.asarray(y1-y),axis=-1).max(),np.sqrt(2)*gap+2e-6)

    def test_upper_edges_and_uncovered_anchor(self):
        s=np.tile([8.5,3.5],4).astype(np.float32)[None]
        anchor=np.array([[[9.,3.5]]],np.float32)
        lo,hi,valid,_=convex_box(s,anchor)
        self.assertTrue(np.asarray(valid).all());self.assertEqual(float(hi[0,0,0]),9)
        bad=np.array([[[8.5,2.5]]],np.float32)
        self.assertFalse(np.asarray(convex_box(s,bad)[2]).any())

    def test_emitted_energy_score_with_atoms(self):
        sample=jnp.array([[[0.,0.],[0.,0.],[2.,0.],[2.,0.]]])
        # E-distance empirical mean=1; ordered U pair mean=4/3, half=2/3.
        self.assertAlmostEqual(float(energy_score(sample,jnp.zeros((1,2)))[0]),1/3,places=6)
        with self.assertRaises(ValueError):energy_score(sample[:,:1],jnp.zeros((1,2)))


if __name__=='__main__':unittest.main()
