"""Focused neural substitution checks; no oracle-based architecture tuning."""
import unittest

import numpy as np
import torch

from ett import finite_crl as b
from ett.finite_neural import NeuralCritic, fit_neural, nce_loss, parameter_sha


class NeuralTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.theta = np.array([[.69, -.08], [.81, .11]])

    def test_dot_product_shape_and_unmasked_decode(self):
        model = NeuralCritic(130)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 944)
        with torch.no_grad():
            logits = model()
            expected = model.phi(model.features) @ model.psi.weight.T
            torch.testing.assert_close(logits.reshape(32, 4), expected)
        q = model.q_values()
        self.assertEqual(q.shape, (5, 4, 2, 4))
        self.assertTrue(np.all(q[0] == 0))
        self.assertTrue(np.all(q[1:] > 0))  # No state/goal/death-value hardcoding.
        np.testing.assert_allclose(q[1:], (1 - .9 ** np.arange(1, 5))[:, None, None, None] * 31 * .25 * np.exp(logits.numpy()))

    def test_nce_class_prior_and_geometric_mass(self):
        counts = np.random.default_rng(491).integers(1, 30, size=(5, 4, 2, 4))
        counts[0] = 0
        p = torch.tensor(counts[1:] / counts[1:].sum(-1, keepdims=True))
        logits = torch.tensor(np.log(b.fit_nce(counts)[1:]), requires_grad=True)
        loss = nce_loss(logits, p)
        loss.backward()
        torch.testing.assert_close(logits.grad, torch.zeros_like(logits), atol=1e-16, rtol=0)
        self.assertAlmostEqual(float(loss.detach()), b.empirical_nce_loss(counts, b.fit_nce(counts)), places=14)

    def test_smoke_reproducibility_and_frozen_critic_update(self):
        counts, _ = b.mc_positives(self.theta, np.random.default_rng(911), 32)
        models = [NeuralCritic(392), NeuralCritic(392)]
        for model in models:
            initial_sha = parameter_sha(model)
            fit = fit_neural(model, torch.optim.Adam(model.parameters(), lr=.01), counts, 32)
            self.assertLess(fit['final_loss'], fit['first_loss'])
            self.assertNotEqual(initial_sha, parameter_sha(model))
            before = parameter_sha(model)
            q = model.q_values()
            paths, _, _ = b.sample_paths(self.theta, np.random.default_rng(199), np.zeros(64, int), None, 4)
            visits = np.stack([np.bincount(paths[:, t], minlength=4) / 64 for t in range(4)])
            _, _, _, _, grad = b.loss_gradient(self.theta, q, visits, 4.)
            b.assert_feasible(b.project(self.theta - .12 * grad))
            self.assertEqual(parameter_sha(model), before)
            self.assertTrue(all(not p.requires_grad for p in model.parameters()))
        np.testing.assert_array_equal(models[0].q_values(), models[1].q_values())

    def test_common_rng_not_identical_outcomes(self):
        theta2 = self.theta.copy()
        theta2[:, 1] = -.15
        rng1, rng2 = np.random.default_rng(512), np.random.default_rng(512)
        left, _ = b.mc_positives(self.theta, rng1, 128)
        right, _ = b.mc_positives(theta2, rng2, 128)
        self.assertEqual(rng1.bit_generator.state, rng2.bit_generator.state)
        self.assertTrue(np.any(left != right))


if __name__ == '__main__':
    unittest.main()
