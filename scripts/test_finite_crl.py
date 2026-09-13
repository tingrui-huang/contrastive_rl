"""Focused checks for the isolated finite-state loss-validation experiment."""
import unittest

import numpy as np

from ett import finite_crl as f
from ett.finite_crl_eval import certificate, exact_values, exact_visits, objective


class FiniteCRLTests(unittest.TestCase):
    def setUp(self):
        self.theta = np.array([[.69, -.08], [.81, .11]])

    def test_full_pair_constraints_and_shared_intervention(self):
        k = f.kernel(self.theta)
        f.assert_feasible(self.theta)
        for s in range(4):
            for xp in range(2):
                for a in range(2):
                    for b in range(2):
                        tv = np.abs(k[s, a, xp] - k[s, b, xp]).sum() / 2
                        self.assertLessEqual(tv, f.L * abs(a - b) + 1e-15)
        p = f.intervention(self.theta)
        np.testing.assert_allclose(p, .75 * k[:, :, 0] + .25 * k[:, :, 1])
        for s in (f.S, f.M):
            np.testing.assert_allclose(f.PI @ p[s, :, s + 1], self.theta[s, 0] + .5 * self.theta[s, 1])
            # Shared slope requires opposite off-diagonal changes.
            np.testing.assert_allclose(k[s, 0, 1, s + 1] + k[s, 1, 0, s + 1], 2 * self.theta[s, 0])

    def test_replay_action_alignment_absorption_reward_horizon(self):
        states = np.tile(np.arange(4), 256)
        actions = np.tile([0, 1], 512)
        left = f.sample_paths(self.theta, np.random.default_rng(918), states, actions, f.H)
        right = f.sample_paths(self.theta, np.random.default_rng(918), states, actions, f.H)
        for a, b in zip(left, right):
            np.testing.assert_array_equal(a, b)
        paths, executed, xp = left
        self.assertEqual(paths.shape, (1024, 5))
        np.testing.assert_array_equal(executed[:, 0], actions)
        self.assertTrue(np.any(xp[:, 0] != executed[:, 0]))
        for state in (f.G, f.D):
            self.assertTrue(np.all(paths[states == state] == state))
        q = exact_values(self.theta)
        self.assertTrue(np.all(q[0] == 0))
        self.assertTrue(np.all(q[1, f.S, :, f.G] == 0))
        np.testing.assert_allclose(q[1, f.M, :, f.G], .1 * f.intervention(self.theta)[f.M, :, f.G])
        np.testing.assert_allclose(q[f.H, f.G, :, f.G], 1 - .9 ** 4)
        self.assertTrue(np.all(q[:, f.D, :, f.G] == 0))

    def test_nce_weighting_stationarity_horizon_and_alpha(self):
        rng = np.random.default_rng(981)
        counts = rng.integers(1, 100, (5, 4, 2, 4))
        counts[0] = 0
        p = counts[1:] / counts[1:].sum(-1, keepdims=True)
        for alpha in (0., .5):
            odds = f.fit_nce(counts, alpha)
            q = f.decode(odds, alpha)
            np.testing.assert_allclose(q[1:], (1 - .9 ** np.arange(1, 5))[:, None, None, None] * p)
            # Expected derivative of the B-by-B binary objective is zero.
            derivative = (-p + (f.B - 1) * f.negative(alpha) * odds[1:]) / (1 + odds[1:]) / f.B
            np.testing.assert_allclose(derivative, 0., atol=1e-16)
        odds0 = f.fit_nce(counts, 0.)
        odds5 = f.fit_nce(counts, .5)
        np.testing.assert_allclose(f.decode(odds0), f.decode(odds5, .5))
        np.testing.assert_allclose(f.decode(odds5)[..., f.G], 2 * f.decode(odds0)[..., f.G])
        # A mean logit produces a geometric mean, not expected Q.
        self.assertGreater(abs(f.PI @ np.array([.1, .9]) - np.exp(f.PI @ np.log([.1, .9]))), .05)

    def test_sampled_mc_positives_match_model_without_exact_training_targets(self):
        counts, steps = f.mc_positives(self.theta, np.random.default_rng(298), 4096)
        self.assertEqual(steps, 8 * 4096 * 10)
        self.assertTrue(np.all(counts[1:].sum(-1) == 4096))
        estimate = f.decode(f.fit_nce(counts))
        np.testing.assert_allclose(estimate, exact_values(self.theta), atol=.012, rtol=0)

    def test_enumerated_gradient_matches_full_return_finite_difference(self):
        truth = exact_values(self.theta)
        visits = exact_visits(self.theta)[:-1]
        dl, off, dg, og, combined = f.loss_gradient(self.theta, truth, visits, 4.)
        numerical = np.zeros((2, 2))
        for index in np.ndindex(2, 2):
            hi, lo = self.theta.copy(), self.theta.copy()
            hi[index] += 1e-6
            lo[index] -= 1e-6
            numerical[index] = (objective(hi) - objective(lo)) / 2e-6
        np.testing.assert_allclose(og, numerical, atol=1e-10, rtol=0)
        self.assertTrue(np.all(dg[:, 0] != 0))
        self.assertTrue(np.all(dg[:, 1] == 0))
        self.assertTrue(np.all(og > 0))
        np.testing.assert_allclose(combined, dg + 4 * numerical, atol=4e-10)
        self.assertTrue(np.isfinite([dl, off]).all())

    def test_box_vertices_and_rational_certificate(self):
        cert = certificate()
        for bits in np.ndindex(2, 2, 2, 2):
            theta = np.column_stack((f.OBSERVED + (2 * np.array(bits[:2]) - 1) * f.EPS,
                                     (2 * np.array(bits[2:]) - 1) * f.L))
            f.assert_feasible(theta)
            self.assertGreaterEqual(objective(theta) + 1e-15, cert['J_opt'])
        projected = f.project(np.array([[-10., 12.], [11., -10.]]))
        f.assert_feasible(projected)
        self.assertEqual(cert['representation_gap_within_declared_family'], 0)

    def test_two_round_training_smoke_without_oracle(self):
        theta = self.theta.copy()
        rng = np.random.default_rng(8991)
        for _ in range(2):
            counts, _ = f.mc_positives(theta, rng, 64)
            odds = f.fit_nce(counts)
            nce = f.empirical_nce_loss(counts, odds)
            self.assertTrue(np.isfinite(nce) and nce > 0)
            paths, _, _ = f.sample_paths(theta, rng, np.full(256, f.S), None, f.H)
            visits = np.stack([np.bincount(paths[:, t], minlength=4) / len(paths) for t in range(f.H)])
            _, _, _, _, gradient = f.loss_gradient(theta, f.decode(odds), visits, 4.)
            theta = f.project(theta - .12 * gradient)
            f.assert_feasible(theta)
        self.assertTrue(np.all(theta != self.theta))


if __name__ == '__main__':
    unittest.main()
