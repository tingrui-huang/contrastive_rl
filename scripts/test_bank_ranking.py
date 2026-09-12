"""Focused contracts for the frozen recorded-outcome distance audit."""
import unittest
import numpy as np

from ett.bank_ranking import (reconstruct_death, outcome_labels, nearest_scores,
    binary_metrics, match_outcomes, pair_ranking)
from ett.run_bank_ranking import PROTOCOL


class RankingTests(unittest.TestCase):
    def test_fatal_landing_and_next_state_age(self):
        xy = np.array([[1., 3.5], [2.2, 3.5], [3.1, 3.5], [3.1, 3.5], [3.1, 3.5], [3.1, 3.5]])
        obs = np.zeros((1, len(xy), 16), np.float32)
        for t in range(len(xy)):
            obs[0, t, :8] = np.concatenate([xy[max(0, t-j)] for j in range(4)])
        bits = np.zeros((1, len(xy), 3), bool); bits[0, 1, 0] = True
        death = reconstruct_death(obs, bits, [True], np.array([6]))
        np.testing.assert_array_equal(death, [2])
        dead, age = outcome_labels(np.zeros(6, int), np.arange(6), death)
        np.testing.assert_array_equal(dead, [False, False, True, True, True, True])
        np.testing.assert_array_equal(age, [-1, -1, 0, 1, 2, 3])
        with self.assertRaises(ValueError):
            reconstruct_death(obs, bits, [False], np.array([6]))
        bad = obs.copy(); bad[0, 3, 2] += .1
        with self.assertRaises(ValueError):
            reconstruct_death(bad, bits, [True], np.array([6]))

    def test_horizon_and_alive_stationarity(self):
        obs = np.tile(np.array([1., 3.5]*8, np.float32), (2, 4, 1))
        bits = np.zeros((2, 4, 3), bool)
        death = reconstruct_death(obs, bits, [False, False], np.array([4, 4]))
        np.testing.assert_array_equal(death, [-1, -1])
        dead, age = outcome_labels(np.array([0, 0]), np.array([2, 3]), np.array([3]))
        np.testing.assert_array_equal(age, [-1, 0])
        self.assertTrue(dead[-1])  # The terminal NEXT row retains its actual onset.

    def test_existing_score_and_same_reference_decomposition(self):
        y = np.zeros((2, 8), np.float32)
        y[1] = .25
        f = np.zeros((2, 8), np.float32)
        f[0, :2] = 3.; f[1, 2:] = 2.
        std = np.linspace(.5, 2, 8).astype(np.float32)
        scores = nearest_scores(y, f, np.ones(8, np.float32)*14, std)
        difference = ((y[:, None]-f)/std)**2
        d = difference[:, :, :2].sum(-1)+difference[:, :, 2:].sum(-1)
        index = d.argmin(1)
        np.testing.assert_array_equal(scores['index'], index)
        np.testing.assert_allclose(scores['distance2'], d[np.arange(2), index], rtol=2e-7)
        np.testing.assert_array_equal(scores['distance2'], scores['newest_distance2']+scores['history_distance2'])
        self.assertGreater(scores['distance2'][0], difference[0, :, :2].sum(-1).min()+difference[0, :, 2:].sum(-1).min())

    def test_bank_tie_order_and_exact_member(self):
        f = np.ones((3, 8), np.float32)
        result = nearest_scores(f[:1], f, np.zeros(8, np.float32), np.ones(8, np.float32))
        self.assertEqual(result['distance2'][0], 0)
        self.assertEqual(result['index'][0], 0)
        self.assertEqual(result['exact_tie_count'][0], 3)

    def test_matching_context_and_episode_reuse(self):
        ep = np.arange(6)
        row = np.array([5, 6, 7, 6, 6, 6])
        state = np.tile(np.array([3.2, 3.5]*4), (6, 1))
        goal = np.ones((6, 8)); source = np.array([0, 0, 0, 0, 1, 0])
        state[5, :2] = [4.2, 3.5]
        failed = np.array([True, True, False, False, False, False])
        pairs, summary = match_outcomes(failed, ~failed, ep, row, state, state, source, goal, PROTOCOL['matching'])
        self.assertEqual(len(pairs), 2)
        self.assertEqual(len(np.unique(ep[pairs])), 4)
        self.assertTrue(np.all(source[pairs[:, 0]]==source[pairs[:, 1]]))
        again, _ = match_outcomes(failed, ~failed, ep, row, state, state, source, goal, PROTOCOL['matching'])
        np.testing.assert_array_equal(pairs, again)
        self.assertEqual(summary['coverage'], 1.)

    def test_metric_direction_ties_and_pair_blocks(self):
        y = np.array([True, True, False, False])
        self.assertEqual(binary_metrics(y, np.array([0, 1, 2, 3]))['roc_auc'], 1.)
        self.assertEqual(binary_metrics(y, np.array([3, 2, 1, 0]))['roc_auc'], 0.)
        self.assertEqual(binary_metrics(y, np.ones(4))['roc_auc'], .5)
        self.assertEqual(binary_metrics(y, np.ones(4))['average_precision'], .5)
        self.assertIsNone(binary_metrics([True], np.array([0]))['roc_auc'])
        r = pair_ranking(np.array([[0, 2], [1, 3]]), np.ones(4))
        self.assertEqual(r['accuracy_half_ties'], .5)
        self.assertEqual(r['ties'], 1.)
        self.assertEqual(r['interval95'], [.5, .5])

    def test_exclude_original_bank_sources_not_only_retained(self):
        validation = np.array([1, 3, 5, 7])
        original = np.array([0, 1, 2, 3])
        retained = np.array([0, 2])
        evaluation = np.setdiff1d(validation, original)
        np.testing.assert_array_equal(evaluation, [5, 7])
        self.assertFalse(np.isin(evaluation, original).any())
        self.assertEqual(len(np.setdiff1d(validation, retained)), 4)


if __name__=='__main__':
    unittest.main()
