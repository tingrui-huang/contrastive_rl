"""Controlled update algebra, estimator isolation and paired uncertainty."""
import unittest
from unittest.mock import patch

import numpy as np
import torch

from ett import pointmaze_update_reference as p
from ett import eval_pointmaze_update_reference as e
from ett import pointmaze_early_pilot as early
from ett import pointmaze_future_average as average
from ett import pointmaze_region_pilot as old
from scripts.test_pointmaze_phase_sampling import fixture


class UpdateReferenceTests(unittest.TestCase):
    def test_update_is_existing_paired_finite_difference_and_block_caps(self):
        rng=np.random.default_rng(55); directions=rng.normal(size=(4, 48))
        sigma=np.r_[np.full(16, .01), np.full(32, .1)]
        diag=np.r_[np.full(16, .5), np.zeros(32)]; off=np.linspace(-.3, .3, 48)
        signed=np.array([[[diag@(sigma*u), off@(sigma*u)],
                          [diag@(-sigma*u), off@(-sigma*u)]] for u in directions])
        theta=np.zeros(48, np.float32)
        proposed, info=p.propose(theta, directions, signed, True)
        expected=np.array([sum(((g@(sigma*u))*u/sigma for u in directions))/4 for g in [diag, off]])
        expected[0, 16:]=0
        np.testing.assert_allclose(info['diagonal_gradient'], expected[0])
        np.testing.assert_allclose(info['surrogate_gradient'], expected[1])
        self.assertLessEqual(np.linalg.norm(proposed[:16]), .03000001)
        self.assertLessEqual(np.linalg.norm(proposed[16:]), .10000001)
        control, _=p.propose(theta, directions, signed, False)
        np.testing.assert_array_equal(control[16:], 0)
        self.assertTrue(p.guard_accept(.12, .1)); self.assertFalse(p.guard_accept(.120001, .1))

    def test_refresh_preserves_averaged_objective_and_adam_continuity(self):
        torch.set_num_threads(1); records, data=fixture()
        context=dict(mean=np.zeros(8), std=np.ones(8))
        a=early.Critic(7); b=early.Critic(7)
        opt=torch.optim.Adam(a.parameters(), lr=.003)
        p.refresh(a, opt, records, context, 18, steps=4)
        average.fit(b, data, average.future_probabilities(records), 'averaged', 18, steps=4)
        self.assertEqual(old.parameter_sha(a), old.parameter_sha(b))
        p.refresh(a, opt, records, context, 19, steps=4)
        self.assertTrue(all(float(state['step'])==8 for state in opt.state.values()))
        self.assertFalse(any(param.requires_grad for param in a.parameters()))

    def test_candidate_first_then_arm_preupdate_not_initial(self):
        class Nominal:
            def sample(self, s, key, count, goal): return np.zeros((len(s), 2), np.float32)
        class Fake:
            nominal=Nominal()
            def __init__(self): self.first=[]; self.continued=[]
            def sample(self, theta, s, a, xp, key, count):
                self.first.append(theta.copy())
                return np.broadcast_to(old.GOAL, (len(s), 1, 8)).copy(), {}
            def actor_jit(self, s, key): return np.zeros((len(s), 2), np.float32)
            def rollout(self, theta, s, key, horizon, first):
                self.continued.append((theta.copy(), horizon))
                return dict(action=np.repeat(first[:, None], horizon, 1), reward=np.ones((len(s), horizon)))
        class Counter:
            def __init__(self): self.count=0
            def add(self, n, purpose): self.count+=n
        class ExactCritic:
            def predict(self, s, a, h, mean, std): return 1-.95**h
        queries=dict(state=np.zeros((2, 8), np.float32), action=np.zeros((2, 2), np.float32),
            h=np.array([1, 3]), weight=np.array([4., 7.]))
        candidate=np.full(48, 2., np.float32); frozen=np.full(48, 1., np.float32)
        counter=Counter(); engine=Fake(); context=dict(mean=np.zeros(8), std=np.ones(8))
        with patch.object(old, 'validate_selected_set'):
            mc, raw=p.surrogate(engine, candidate, frozen, queries, None, context, 90, counter)
            predicted, _=p.surrogate(engine, candidate, frozen, queries, ExactCritic(), context, 90, counter)
        self.assertAlmostEqual(mc, predicted, places=7)
        self.assertEqual(len(engine.continued), 1)
        np.testing.assert_array_equal(engine.continued[0][0], frozen)
        self.assertEqual(engine.continued[0][1], 2)
        np.testing.assert_array_equal(engine.first[0], candidate)
        self.assertEqual(counter.count, 8+16*2+8)
        np.testing.assert_array_equal(raw['q'][0], 0)
        self.assertTrue(np.isnan(raw['continuation_rewards'][0]).all())

    def test_paired_visitation_indices_horizons_and_weights(self):
        records, _=fixture(); a=p.visitation(records, 63); b=p.visitation(records, 63)
        for key in a: np.testing.assert_array_equal(a[key], b[key])
        for i, (path, t) in enumerate(zip(a['path'], a['t'])):
            r=records[path]; h=r['action'].shape[1]
            self.assertEqual(a['h'][i], h-t); self.assertAlmostEqual(a['weight'][i], h*.95**t)
            np.testing.assert_array_equal(a['state'][i], r['states'][0, t])
            np.testing.assert_array_equal(a['action'][i], r['action'][0, t])

    def test_paired_uncertainty_and_inconclusive_rules(self):
        groups=np.repeat(np.arange(3), 12); rng=np.random.default_rng(3)
        noise=rng.normal(size=(36, 64)); difference=(noise-.01)-noise
        row=e.paired_returns(difference, groups)
        self.assertTrue(e.below_zero(row)); self.assertLess(row['conditional_mc_se'], 1e-15)
        self.assertFalse(e.below_zero(e.paired_returns(np.zeros((36, 64)), groups)))
        uncertain=np.repeat(np.linspace(-.1, .1, 36)[:, None], 64, 1)
        row=e.paired_returns(uncertain, groups)
        self.assertFalse(e.below_zero(row)); self.assertFalse(e.above_zero(row))
        clustered=e.diagonal_interval(np.zeros(512), np.arange(512)//4)
        self.assertEqual(clustered['episodes'], 128)
        np.testing.assert_array_equal(clustered['episode_ci95'], [0., 0.])


if __name__=='__main__': unittest.main()
