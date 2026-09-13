"""Bounded correctness checks, including old benchmark regressions elsewhere."""
import unittest
import numpy as np
import torch
from ett import finite_setback as b
from ett.finite_setback_eval import certificate, exact, objective, visits


class SetbackTests(unittest.TestCase):
    def setUp(self):
        self.theta = np.array([.395, .356, .02, -.01])
        torch.set_num_threads(1)

    def test_full_pairs_hex_projection_and_shared_cells(self):
        for theta in (self.theta, b.project(np.array([-10., 10., 100., -50.]))):
            b.feasibility(theta)
            k = b.kernel(theta)
            for s in range(b.N):
                for xp in range(2):
                    for a in range(2):
                        for c in range(2):
                            self.assertLessEqual(.5 * abs(k[s, a, xp] - k[s, c, xp]).sum(), .15 * abs(a - c) + 1e-15)
            np.testing.assert_allclose(b.intervention(theta), .75 * k[:, :, 0] + .25 * k[:, :, 1])
            np.testing.assert_allclose(k[b.S, 1, 0] + k[b.S, 0, 1], 2 * k[b.S, 0, 0], atol=1e-15)
        # Verify the convex projection variational inequality against vertices.
        vertices = .15 * np.array([[1, 0], [1, -1], [0, -1], [-1, 0], [-1, 1], [0, 1]])
        for z in np.random.default_rng(204).normal(size=(32, 2)):
            p = b.project_hex(z, .15)
            self.assertLessEqual(np.max((vertices - p) @ (z - p)), 1e-14)

    def test_recovery_rewards_ties_horizon_replay(self):
        states = np.array([b.R, b.D, b.P])
        left = b.sample_paths(self.theta, np.random.default_rng(551), states, np.array([1, 0, 1]), 4)
        right = b.sample_paths(self.theta, np.random.default_rng(551), states, np.array([1, 0, 1]), 4)
        for x, y in zip(left, right):
            np.testing.assert_array_equal(x, y)
        np.testing.assert_array_equal(left[0][0], [b.R, b.P, b.G, b.G, b.G])
        np.testing.assert_array_equal(left[0][1], [b.D] * 5)
        np.testing.assert_array_equal(left[1][:, 0], [1, 0, 1])
        self.assertFalse(b.R == b.G or b.D == b.G)
        q = exact(self.theta)
        np.testing.assert_allclose(q[3, b.R, :, b.G], .171, atol=1e-15)
        self.assertTrue(np.all(q[:, b.D, :, b.G] == 0))
        self.assertTrue(np.all(q[1, b.R, :, b.G] == 0))
        np.testing.assert_allclose(q[4, b.G, :, b.G], 1 - .9 ** 4)

    def test_certificate_does_not_replace_reward_by_death(self):
        c = certificate()
        self.assertAlmostEqual(c['J_opt'], .1282545)
        self.assertGreater(c['alternative_J'], c['J_opt'])
        a = b.PI @ b.intervention(c['witness'])[b.S]
        d = b.PI @ b.intervention(c['maximum_death_alternative'])[b.S]
        self.assertAlmostEqual(a[b.D], d[b.D])
        self.assertGreater(a[b.R], d[b.R])

    def test_exact_categorical_gradient(self):
        q = exact(self.theta)
        og = b.gradient(self.theta, q, visits(self.theta)[:-1], 4.)[3]
        numerical = []
        for i in range(4):
            plus, minus = self.theta.copy(), self.theta.copy()
            plus[i] += 1e-6; minus[i] -= 1e-6
            numerical.append((objective(plus) - objective(minus)) / 2e-6)
        np.testing.assert_allclose(og, numerical, atol=1e-10, rtol=0)
        np.testing.assert_allclose(og, [.2439, .1539, .12195, .07695], atol=1e-15)

    def test_neural_nce_calibration_and_smoke(self):
        counts, steps = b.mc_counts(self.theta, np.random.default_rng(993), 64)
        self.assertEqual(steps, 6400)
        self.assertTrue(np.all(counts[1:].sum(-1) == 64))
        model = b.Critic(114)
        q0 = model.values()
        self.assertTrue(np.all(q0[0] == 0) and np.all(q0[1:] > 0))
        opt = torch.optim.Adam(model.parameters(), lr=.01)
        loss = b.fit(model, opt, counts, 32)
        self.assertTrue(np.isfinite(loss))
        self.assertTrue(all(not p.requires_grad for p in model.parameters()))
        rng = np.random.default_rng(718)
        positive = rng.integers(1, 30, size=(4, 5, 2, 5)).astype(float)
        positive /= positive.sum(-1, keepdims=True)
        logits = torch.tensor(np.log(positive * 5 / 31), requires_grad=True)
        b.nce_loss(logits, torch.tensor(positive)).backward()
        torch.testing.assert_close(logits.grad, torch.zeros_like(logits), atol=1e-16, rtol=0)


if __name__ == '__main__':
    unittest.main()
