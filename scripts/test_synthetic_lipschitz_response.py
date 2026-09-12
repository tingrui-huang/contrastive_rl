"""Focused tests of the conditional spline construction, not a sampled proof."""
import unittest

import numpy as np
import torch

from scripts.synthetic_shared_response import make_data, two_losses
from scripts.synthetic_lipschitz_response import ConditionalSpline, exact_interval_diagnostics, initialize


class ConditionalSplineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_arbitrary_pairs_after_large_parameter_changes(self):
        model = initialize(2)
        rng = np.random.default_rng(17)
        triples = rng.uniform(-1,1,(5000,3))
        left,right = torch.from_numpy(triples[:,[0,2]]),torch.from_numpy(triples[:,[1,2]])
        for magnitude in [0., .1, 1., 10.]:
            with torch.no_grad():
                for p in model.parameters():
                    p.add_(magnitude*torch.randn_like(p))
                excess = (model(left)-model(right)).abs() - (left[:,0]-right[:,0]).abs()
                # Large intercepts cancel in exact arithmetic; allow evaluation roundoff.
                self.assertLessEqual(float(excess.max()),1e-10)
                _,slopes=model.coefficients(left[:,1])
                self.assertLessEqual(float(slopes.abs().max()),1.)

    def test_target_is_representable_without_using_it_in_initialization(self):
        model = initialize(3)
        with torch.no_grad():
            for p in model.parameters():
                p.zero_()
            model.conditioner[-1].bias[0] = -2.
            model.conditioner[-1].bias[1:9] = 1.
            model.conditioner[-1].bias[9:] = -1.
        pairs = torch.from_numpy(np.random.default_rng(8).uniform(-1,1,(5000,2)))
        torch.testing.assert_close(model(pairs),-(pairs[:,0]-pairs[:,1]).abs(),rtol=0,atol=1e-15)
        metrics,_=exact_interval_diagnostics(model,np.linspace(-1,1,201))
        self.assertEqual(metrics['max_exact_interval_slope'],1.)
        self.assertLess(metrics['max_endpoint_identity_error'],1e-14)

    def test_knot_continuity_and_one_sided_slopes(self):
        model=initialize(9)
        xp=.137
        with torch.no_grad():
            _,slopes=model.coefficients(torch.tensor([xp],dtype=torch.float64))
            for j in range(1,16):
                x=float(model.knots[j])+xp
                if not -1<x<1:
                    continue
                eps=1e-7
                values=model(torch.tensor([[x-eps,xp],[x,xp],[x+eps,xp]],dtype=torch.float64))
                torch.testing.assert_close((values[1]-values[0])/eps,slopes[0,j-1],rtol=0,atol=1e-8)
                torch.testing.assert_close((values[2]-values[1])/eps,slopes[0,j],rtol=0,atol=1e-8)
                self.assertLess(float((values[2]-values[0]).abs()),2.01e-7)

    def test_no_forced_anchor_and_joint_gradient_arithmetic(self):
        model=initialize(4)
        d,o=[torch.from_numpy(a).double() for a in make_data(5,64,256)]
        before=model(d).detach().clone()
        self.assertGreater(float(before.abs().max()),.001)
        with torch.no_grad():
            model.conditioner[-1].bias[0].add_(.2)
        torch.testing.assert_close(model(d),before+.2)
        total,diag,off=two_losses(model,d,o,1.,1.)
        torch.testing.assert_close(total,diag+off)
        params=tuple(model.parameters())
        gd=torch.autograd.grad(diag,params,retain_graph=True)
        go=torch.autograd.grad(off,params,retain_graph=True)
        gt=torch.autograd.grad(total,params)
        for a,b,c in zip(gd,go,gt):
            self.assertGreater(float(a.norm()),0)
            self.assertGreater(float(b.norm()),0)
            torch.testing.assert_close(a+b,c)
        self.assertTrue(all(p.requires_grad for p in params))

    def test_parameter_roundtrip(self):
        import io
        model=initialize(1)
        pairs=torch.from_numpy(make_data(2,32,128)[1])
        stream=io.BytesIO()
        torch.save(model.state_dict(),stream);stream.seek(0)
        copy=ConditionalSpline();copy.load_state_dict(torch.load(stream,weights_only=True))
        torch.testing.assert_close(model(pairs),copy(pairs),rtol=0,atol=0)


if __name__=='__main__':
    unittest.main()
