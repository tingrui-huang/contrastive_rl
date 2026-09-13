"""Analytic checks only: no experiment/model samples or parameter updates."""
import unittest
import numpy as np
from ett import pointmaze_continuation_precision as d
from ett import report_continuation_precision as r

class DiagnosticTests(unittest.TestCase):
    def test_design_recovers_uniform_time_and_visitation(self):
        # Exhaustive synthetic draws in unequal time cells, strongly oversampled t=0.
        H=40;times=[];mass=[];cells=[]
        for b,(lo,hi) in enumerate(d.bounds(H)):
            ts=np.tile(np.arange(lo,hi),120//(hi-lo))
            times.extend(ts);mass.extend(np.full(len(ts),(hi-lo)/(H*len(ts))));cells.extend([b]*len(ts))
        times=np.array(times);a=np.array(mass)
        np.testing.assert_allclose(a@times,np.arange(H).mean())
        np.testing.assert_allclose(a@(H*.95**times),np.sum(.95**np.arange(H)))
        q=dict(mass=a,cell=np.array(cells),weight=H*.95**times)
        fit=r.estimate(times,q)
        self.assertAlmostEqual(fit['mean'],19.5)
        mask=times<4;sub=r.estimate(times,q,mask)
        self.assertAlmostEqual(sub['mean'],1.5)
        self.assertAlmostEqual(sub['target_mass'],.1)

    def test_paired_grouping_and_classification(self):
        q=dict(mass=np.ones(32)/32,cell=np.repeat([0,1],16),weight=np.ones(32))
        rng=np.random.default_rng(0);x=rng.normal(0,.002,(32,8))
        np.testing.assert_allclose(r.estimate(x,q)['mean'],x.mean())
        self.assertEqual(r.estimate(x-x,q)['se'],0)
        rows={k:r.estimate(x,q) for k in ['critic','mc','mc_minus_critic']}
        self.assertEqual(r.classify(rows),'precisely small effect within tolerance')
        rows['mc_minus_critic']=r.estimate(x+.03,q)
        self.assertEqual(r.classify(rows),'meaningful critic error')
        rows={k:r.estimate(x+.03,q) for k in ['critic','mc','mc_minus_critic']}
        rows['mc_minus_critic']=r.estimate(x,q)
        self.assertEqual(r.classify(rows),'agreement at a meaningful effect')
        rows={k:r.estimate(x*100,q) for k in rows}
        self.assertIn('insufficient precision',r.classify(rows))

    def test_budget_and_checkpoint_provenance(self):
        ctx=dict(np.load(d.p.phase.PREVIOUS/'contexts.npz'))
        H=50-ctx['train_indices'][:,1]
        self.assertEqual(2*(64*H.sum()+2304*8*2+2304*8*2*2*48)+256,d.CONFIG['expected'])
        sources,details=d.provenance()
        self.assertEqual(set(details),set(d.CONFIG['pairs']))
        self.assertTrue(sources)

if __name__=='__main__':unittest.main()
