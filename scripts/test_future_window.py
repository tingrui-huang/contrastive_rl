"""Focused checks for coupled future-window sampling and native replay."""
import unittest
import numpy as np
from ett.future_window import future_index, synthetic_batch, batch_stream
from ett.policy_improvement import replay
from scripts.test_policy_improvement import fixture
from ett.eval_policy_improvement import native


class FutureWindowTests(unittest.TestCase):
    def test_geometric_distribution(self):
        rng = np.random.default_rng(41000001)
        for i in [0, 1, 2]:
            for h in [3, 10, 50]:
                j = future_index(i, h, rng.random(100000))
                expected = .95**np.arange(1, h-i+1)
                expected /= expected.sum()
                observed = np.bincount(j-i, minlength=h-i+1)[1:]/len(j)
                np.testing.assert_allclose(observed, expected, atol=.006)
                self.assertTrue(np.all((j > i) & (j <= h)))
        with self.assertRaises(ValueError):
            future_index(2, 2, .5)

    def test_length_independent_anchors_padding_and_goals(self):
        obs, actions = fixture(48, 50)
        states = obs[:, :, :8].copy()
        horizons = np.arange(3, 51)
        for e, h in enumerate(horizons):
            states[e, h+1:] = np.nan
            actions[e, h:] = np.nan
        short, si = synthetic_batch(states, actions, horizons, 50000, False,
            np.random.default_rng(1), np.random.default_rng(2))
        long, li = synthetic_batch(states, actions, horizons, 50000, True,
            np.random.default_rng(1), np.random.default_rng(2))
        for key in ['episode', 'time', 'horizon']:
            np.testing.assert_array_equal(si[key], li[key])
        np.testing.assert_array_equal(short.action, long.action)
        np.testing.assert_array_equal(short.observation[:, :8], long.observation[:, :8])
        np.testing.assert_array_equal(short.next_observation[:, :8], long.next_observation[:, :8])
        np.testing.assert_array_equal(long.observation[:, 8:], states[li['episode'], li['future']])
        self.assertTrue(np.isfinite(long.observation).all())
        self.assertTrue(np.any(li['future'] > 3))
        counts = np.bincount(li['episode'], minlength=48)
        self.assertLess(np.max(np.abs(counts/len(li['episode'])-1/48)), .003)

    def test_separate_rng_streams_and_masks(self):
        obs, action = fixture(12, 50)
        shared = dict(states=np.broadcast_to(obs[:, :, :8], (20, 12, 51, 8)),
                      action=np.broadcast_to(action, (20, 12, 50, 2)), horizon=np.full((20, 12), 50))
        rng = dict(anchor=1, future=2, permutation=3)
        b = batch_stream(replay(obs, action, 5), shared, 'B', 0, rng)
        c = batch_stream(replay(obs, action, 5), shared, 'C', 0, rng)
        count = 0
        for _ in range(105):
            sb, ib = next(b)
            sc, ic = next(c)
            for key in ['source', 'permutation', 'episode', 'time']:
                np.testing.assert_array_equal(ib[key], ic[key])
            for key in ib['offline']:
                np.testing.assert_array_equal(ib['offline'][key], ic['offline'][key])
            np.testing.assert_array_equal(sb.action, sc.action)
            count += ib['count']
        self.assertEqual(count, 105*256//10)

    def test_native_fixed_shape_replay(self):
        actor = lambda params, obs: np.tile(np.array([.8, 0], np.float32), (len(obs), 1))
        seeds = [41000002, 41000003]
        first, second = native(None, actor, seeds), native(None, actor, seeds)
        for field in first:
            np.testing.assert_array_equal(first[field], second[field])
        self.assertEqual(first['reward'].shape, (2, 50))


if __name__ == '__main__':
    unittest.main()
