"""Objective equivalence, rollout isolation, and uncertainty regression tests."""
import unittest
import numpy as np
import torch
from torch.nn import functional as F

from ett import pointmaze_future_average as p
from ett import eval_pointmaze_future_average as e
from ett import pointmaze_phase_sampling as prior
from ett import pointmaze_early_pilot as early
from ett import pointmaze_region_pilot as old
from scripts.test_pointmaze_phase_sampling import fixture


class FutureAverageTests(unittest.TestCase):
    def test_suffix_expectation_loss_and_gradient(self):
        rewards=np.array([[0., 1., 0., 1.], [1., 0., 0., 0.]])
        probabilities=p.future_probabilities([{'reward':rewards}]).reshape(2, 4)
        for t in range(4):
            weights=.95**np.arange(4-t); weights/=weights.sum()
            np.testing.assert_allclose(probabilities[:, t], rewards[:, t:]@weights)
            logits=torch.tensor([[.7, -.3], [-2., .9]], dtype=torch.float64, requires_grad=True)
            loss=p.nce(logits, torch.tensor([0, 1]), torch.from_numpy(probabilities[:, t]), 'averaged')
            expected=sum(float(w)*p.nce(logits, torch.from_numpy(rewards[:, t+k].astype(np.int64)),
                torch.zeros(2), 'sampled') for k, w in enumerate(weights))
            torch.testing.assert_close(loss, expected)
            a=torch.autograd.grad(loss, logits, retain_graph=True)[0]
            b=torch.autograd.grad(expected, logits)[0]
            torch.testing.assert_close(a, b)
        np.testing.assert_array_equal(probabilities[:, -1], rewards[:, -1])

    def test_original_uniform_baseline_and_paired_rows(self):
        torch.set_num_threads(1); records, data=fixture(); probability=p.future_probabilities(records)
        a=early.Critic(123); b=early.Critic(123); alternative=early.Critic(123)
        _, ai=p.fit(a, data, probability, 'sampled', 124, steps=4)
        prior.fit_rows(b, data, 'uniform', 124, steps=4)
        _, bi=p.fit(alternative, data, probability, 'averaged', 124, steps=4)
        self.assertEqual(old.parameter_sha(a), old.parameter_sha(b))
        self.assertEqual(ai['row_stream_sha256'], bi['row_stream_sha256'])
        np.testing.assert_array_equal(ai['exposures'], bi['exposures'])

    def test_only_base_continues_and_terminal_has_no_call(self):
        class Fake:
            def __init__(self): self.calls=[]
            def rollout(self, theta, ns, key, horizon, first):
                self.calls.append((theta.copy(), horizon, first.copy()))
                return dict(action=np.repeat(first[:, None], horizon, axis=1),
                    reward=np.ones((len(ns), horizon)))
        class Ledger:
            def __init__(self): self.total=0
            def add(self, n, purpose): self.total+=n
        engine=Fake(); ledger=Ledger(); s=np.zeros((3, 8)); action=np.ones((3, 2))*.2
        value, _=e.continuation(engine, s, action, 3, None, ledger)
        np.testing.assert_allclose(value, 1-.95**3)
        np.testing.assert_array_equal(engine.calls[0][0], np.zeros(48))
        np.testing.assert_array_equal(engine.calls[0][2], action)
        value, reward=e.continuation(engine, s, action, 0, None, ledger)
        np.testing.assert_array_equal(value, 0); self.assertEqual(reward.shape, (3, 0))
        self.assertEqual(len(engine.calls), 1); self.assertEqual(ledger.total, 9)

    def test_queries_match_exact_horizon_and_visitation_weight(self):
        records, _=fixture(); queries=e.select_queries(records, 4)
        for root, record in enumerate(records):
            t=queries['t'][root]; h=record['action'].shape[1]
            np.testing.assert_array_equal(queries['state'][root], record['states'][0, t])
            np.testing.assert_array_equal(queries['action'][root], record['action'][0, t])
            self.assertEqual(queries['h'][root], h-t)
            self.assertAlmostEqual(queries['weight'][root], h*.95**t)

    def test_feasible_offsets_and_unresolved_differences(self):
        for name, theta in p.candidates().items():
            np.testing.assert_array_equal(theta[:16], np.zeros(16))
            self.assertAlmostEqual(float(np.linalg.norm(theta)), 0 if name=='base' else .1)
            self.assertLessEqual(float(np.linalg.svd(np.asarray(old.matrices(theta[16:])), compute_uv=False).max()), 1.)
        idx=e.boot_indices(np.repeat(np.arange(3), 12))
        resolved=e.interval(np.ones((36, 128))*.01, idx)
        self.assertTrue(e.resolved(resolved, .001))
        self.assertFalse(e.resolved(e.interval(np.zeros((36, 128)), idx), .001))
        noisy=np.tile(np.r_[np.ones(64), -np.ones(64)+.01], (36, 1))
        self.assertFalse(e.resolved(e.interval(noisy, idx), .001))
        # Shared baseline noise cancels before estimating a difference's uncertainty.
        rng=np.random.default_rng(50); shared=rng.normal(size=(36, 128))
        paired=e.interval((shared+.01)-shared, idx)
        self.assertLess(paired['conditional_mc_se'], 1e-15)


if __name__=='__main__': unittest.main()
