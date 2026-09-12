"""Numerical checks for the isolated scalar response experiment."""
import unittest

import numpy as np
import torch

from scripts.synthetic_shared_response import (CONFIG, SharedResponse, action_norm_bound,
    evaluate, initialize, make_data, sample_off, target, two_losses)


class SyntheticResponseTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_target_anchor_full_bound_and_corner(self):
        rng = np.random.default_rng(7)
        x, z, xp = rng.uniform(-1, 1, (3, 10000))
        for scale in CONFIG['L_values']:
            np.testing.assert_array_equal(target(xp, xp, scale), 0.)
            self.assertTrue(np.all(np.abs(target(x, xp, scale) - target(z, xp, scale)) <= scale * np.abs(x - z) + 1e-14))
            e = 1e-5
            self.assertAlmostEqual((target(0, 0, scale) - target(-e, 0, scale)) / e, scale)
            self.assertAlmostEqual((target(e, 0, scale) - target(0, 0, scale)) / e, -scale)

    def test_sampling_and_reproducibility(self):
        diag, off = make_data(1, 1000, 10000)
        np.testing.assert_array_equal(diag[:, 0], diag[:, 1])
        np.testing.assert_array_equal(off, make_data(1, 1000, 10000)[1])
        self.assertTrue(np.all(np.abs(off) <= 1))
        d = np.abs(off[:, 0] - off[:, 1])
        self.assertTrue(np.all(d > 0))
        self.assertEqual(int(np.sum(d < .1)), 5000)
        self.assertFalse(np.array_equal(off, sample_off(np.random.default_rng(2), 10000)))

    def test_anchored_does_not_imply_full_bound(self):
        # Fixed xp=0: |h(x)-h(0)|<=|x|, but an arbitrary-pair slope exceeds 1.
        x = np.linspace(-1, 1, 10001)
        h = x * np.sin(10 * x)
        self.assertTrue(np.all(np.abs(h) <= np.abs(x) + 1e-15))
        self.assertGreater(float(np.max(np.abs(np.diff(h) / np.diff(x)))), 5.)

    def test_matched_initial_function_and_no_forced_diagonal(self):
        base, signed = initialize(3), initialize(3, True)
        pairs = torch.from_numpy(make_data(7, 100, 200)[1])
        torch.testing.assert_close(base(pairs), signed(pairs), rtol=0, atol=0)
        diag = torch.from_numpy(make_data(7, 100, 200)[0])
        before = base(diag).detach().clone()
        with torch.no_grad():
            base.layers[-1].bias.add_(.25)
        torch.testing.assert_close(base(diag), before + .25)
        self.assertFalse(torch.all(before == 0))

    def test_two_loss_shared_gradients_and_scale(self):
        model = initialize(4)
        diag, off = [torch.from_numpy(a) for a in make_data(3, 64, 128)]
        total, d, o = two_losses(model, diag, off, 4., 1.)
        torch.testing.assert_close(total, d + o)
        params = tuple(model.parameters())
        gd = torch.autograd.grad(d, params, retain_graph=True)
        go = torch.autograd.grad(o, params, retain_graph=True)
        gt = torch.autograd.grad(total, params)
        for a, b, c in zip(gd, go, gt):
            self.assertGreater(float(a.norm()), 0)
            self.assertGreater(float(b.norm()), 0)
            torch.testing.assert_close(a + b, c)
        for scale in [.25, 1., 4.]:
            value, ld, lo = two_losses(model, diag, off, scale, 1.)
            torch.testing.assert_close(value / scale**2, total.detach() / 16)
            control, _, _ = two_losses(model, diag, off, scale, 0.)
            torch.testing.assert_close(control, ld)

    def test_constructive_representation_and_analytical_bound(self):
        model = SharedResponse(width=2)
        with torch.no_grad():
            for p in model.parameters():
                p.zero_()
            model.layers[0].weight.copy_(torch.tensor([[1., -1.], [-1., 1.]]))
            model.layers[2].weight.copy_(torch.eye(2))
            model.layers[4].weight.fill_(-1.)
        pairs = torch.from_numpy(make_data(2, 128, 2048)[1])
        torch.testing.assert_close(model(pairs), -(pairs[:, 0] - pairs[:, 1]).abs())
        # The valid norm product is 2 although the exact action constant is 1.
        self.assertAlmostEqual(action_norm_bound(model), 2.)
        config = dict(CONFIG)
        result, _ = evaluate(model, 4., config, make_data(870001, 4096, 16384))
        self.assertLess(result['off']['normalized_rmse'], 1e-7)
        self.assertEqual(result['arbitrary_random_pairs']['fraction_above_tolerance'], 0.)
        self.assertEqual(result['anchored_learned_value']['fraction_above_tolerance'], 0.)
        self.assertEqual(result['diagonal']['normalized_mse'], 0.)


if __name__ == '__main__':
    unittest.main()
