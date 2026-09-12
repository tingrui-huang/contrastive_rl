"""Focused numerical tests for the predetermined scalar budget experiment."""
import unittest

import numpy as np
import torch

from scripts.synthetic_failure_bank import CONFIG, bank_losses, initialize, sample_data, train
from scripts.synthetic_budget_reference import (population_reference, optimal_diagonal,
    fixed_class_reference, diagonal_extrema)
from scripts.synthetic_failure_reference import population_reference as old_reference, quadrature


class BudgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_lambda_reference_and_regime(self):
        for weight in [0., .004, .1, .9]:
            ref = population_reference(weight)
            self.assertAlmostEqual(ref['max_abs_diagonal'], abs(float(optimal_diagonal(0., weight))))
            z, w = np.polynomial.legendre.leggauss(128)
            mean_distance = (1+z*z)/2
            d = optimal_diagonal(z, weight)
            np.testing.assert_allclose(2*d+2*weight*(3+d-mean_distance), 0., atol=1e-14)
            self.assertGreater(np.min(3+d-1-np.abs(z)), 0.)
            self.assertAlmostEqual(float(w@(d*d)/2), ref['diagonal_mse'])
            # Integrate conditional D moments independently of the saved formulas.
            ed2 = 1/3+z*z
            failure = (3+d)**2-2*(3+d)*mean_distance+ed2
            self.assertAlmostEqual(float(w@failure/2), ref['failure_mse'])
        self.assertLess(population_reference(.004)['max_abs_diagonal'], .01)
        self.assertAlmostEqual(population_reference(.1)['total'], old_reference()['total'])
        with self.assertRaises(ValueError):
            population_reference(1.)

    def test_global_conditional_reference(self):
        # Piecewise integration splits at the bank contact points. Probe both branches.
        for weight in [.004, .1]:
            for z in [-1., -.4, 0., .7, 1.]:
                optimum = float(optimal_diagonal(z, weight))
                def objective(d):
                    cuts = sorted(set([-1., 1., z]+[v for v in [z-(3+d), z+(3+d)] if -1<v<1]))
                    integral = 0.
                    for lo, hi in zip(cuts[:-1], cuts[1:]):
                        y = np.maximum(0., 3+d-np.abs(np.array([lo, (lo+hi)/2, hi])-z))**2
                        integral += (hi-lo)*(y[0]+4*y[1]+y[2])/12
                    return d*d+weight*integral
                for d in np.r_[np.linspace(-3, 1, 201), optimum-.0001, optimum+.0001]:
                    self.assertGreaterEqual(objective(d)+1e-14, objective(optimum))

    def test_shared_marginal_and_two_loss_weight(self):
        diagonal, off = sample_data(34, 128, 512)
        np.testing.assert_array_equal(np.repeat(diagonal[:, 1], 4), off[:, 1])
        model = initialize(1)
        d, o = torch.from_numpy(diagonal), torch.from_numpy(off)
        bank = torch.tensor([-3.], dtype=torch.float64, requires_grad=True)
        for weight in [0., .004, .1]:
            total, dl, fl = bank_losses(model, d, o, bank, weight)
            self.assertEqual(float(total.detach()), float((dl+weight*fl).detach()))
            torch.testing.assert_close(fl, (model(o)+3).square().mean())
            total.backward()
            self.assertIsNone(bank.grad)

    def test_initialization_and_batch_schedule_match(self):
        data = sample_data(3, 128, 512)
        config = dict(CONFIG, steps=3, monitor_every=1)
        _, _, old = train('bank_joint', 0, config, data, data)
        _, _, new = train('bank_joint', 0, dict(config, lambda_off=.004), data, data)
        for key in ['initial_state_sha256', 'batch_schedule_sha256', 'data_sha256']:
            self.assertEqual(old[key], new[key])

    def test_feasible_reference_and_extrema(self):
        for weight in [.004, .1]:
            model = fixed_class_reference(weight)
            extrema, arrays = diagonal_extrema(model)
            self.assertAlmostEqual(extrema['max_abs'], population_reference(weight)['max_abs_diagonal'], places=13)
            q = quadrature(model, 256)
            gap = q['diagonal_mse']+weight*q['failure_mse']-population_reference(weight)['total']
            self.assertAlmostEqual(gap, population_reference(weight)['class_reference_population_gap'], places=12)
            self.assertLess(extrema['max_endpoint_reconstruction_error'], 1e-12)
            self.assertEqual(arrays['diagonal_intervals'][0, 0], -1.)
            self.assertEqual(arrays['diagonal_intervals'][-1, 1], 1.)

    def test_extrema_finds_narrow_peak_and_fraction(self):
        model = initialize(0)
        center, width, height = .123456, .0001, .02
        with torch.no_grad():
            for p in model.parameters():
                p.zero_()
            model.conditioner[0].weight[:3, 0] = 1.
            model.conditioner[0].bias[:3] = torch.tensor([-center+width, -center, -center-width], dtype=torch.float64)
            model.conditioner[2].weight.copy_(torch.eye(32, dtype=torch.float64))
            model.conditioner[4].weight[0, :3] = torch.tensor([1., -2., 1.])*height/width
        metrics, _ = diagonal_extrema(model)
        self.assertAlmostEqual(metrics['max_abs'], height, places=10)
        self.assertAlmostEqual(metrics['argmax'], center, places=12)
        self.assertAlmostEqual(metrics['uniform_fraction_exceeding_epsilon'], width/2, places=10)
        grid = torch.linspace(-1, 1, 201, dtype=torch.float64)
        self.assertLess(float(model(torch.stack((grid, grid), -1)).detach().abs().max()), .01)

    def test_extrema_reconstruction_and_structural_bound(self):
        model = initialize(2)
        with torch.no_grad():
            model.conditioner[4].weight.mul_(20)
        _, arrays = diagonal_extrema(model)
        pieces = arrays['diagonal_intervals']
        z = np.random.default_rng(49).uniform(-1, 1, 10003)
        idx = np.searchsorted(pieces[:, 1], z)
        expected = pieces[idx, 2]*z+pieces[idx, 3]
        with torch.no_grad():
            values = model(torch.from_numpy(np.stack((z, z), -1))).numpy()
            triples = np.random.default_rng(87).uniform(-1, 1, (10003, 3))
            left = model(torch.from_numpy(triples[:, [0, 2]])).numpy()
            right = model(torch.from_numpy(triples[:, [1, 2]])).numpy()
            slopes = model.coefficients(torch.from_numpy(z))[1]
        np.testing.assert_allclose(values, expected, atol=1e-12, rtol=0)
        self.assertLessEqual(float(slopes.abs().max()), 1.)
        self.assertLessEqual(np.max(np.abs(left-right)-np.abs(triples[:, 0]-triples[:, 1])), 1e-12)


if __name__=='__main__':
    unittest.main()
