"""Focused replay and native-evaluation contract tests."""
import unittest
import numpy as np
from ett.policy_improvement import replay, mixed_batch, synthetic_count
from ett.fixed_actor_continuation import TimedSimulator
from ett.eval_policy_improvement import native


def fixture(n=12, steps=3):
    states = np.zeros((n, steps+1, 8), np.float32)
    for e in range(n):
        states[e, 0] = np.tile([e/20, 3.5], 4)
        for t in range(steps):
            states[e, t+1] = np.concatenate([states[e, t, :2]+[.01, 0], states[e, t, :6]])
    obs = np.concatenate([states, np.full_like(states, 99)], -1)
    act = np.full((n, steps, 2), .123, np.float32)
    return obs, act


class Contracts(unittest.TestCase):
    def test_goal_boundaries_and_no_auxiliary(self):
        obs, act = fixture()
        buf = replay(obs, act, 1)
        batch, index = buf.sample_audited(10000)
        e, i, j = [index[k] for k in ['episode', 'time', 'future']]
        np.testing.assert_array_equal(batch.observation[:, 8:], obs[e, j, :8])
        np.testing.assert_array_equal(batch.action, act[e, i])
        self.assertTrue(np.all(j > i) and np.all(j <= 3))
        self.assertTrue(np.isnan(batch.next_action).all())
        self.assertFalse(np.any(batch.observation == 99))
        for t in range(3):
            rows = j[i == t]
            expected = .95 ** np.arange(1, 4-t)
            expected /= expected.sum()
            actual = np.array([np.mean(rows == v) for v in range(t+1, 4)])
            np.testing.assert_allclose(actual, expected, atol=.035)

    def test_mixing_and_recent_buffer(self):
        obs, act = fixture()
        offline, synthetic = replay(obs, act, 0), replay(obs, act, 2)
        rng = np.random.default_rng(3)
        counts = []
        for u in range(5):
            batch, audit = mixed_batch(offline, synthetic, u, rng)
            counts.append(audit['source'].sum())
            self.assertEqual(batch.observation.shape, (256, 16))
            self.assertEqual(audit['source'].sum(), synthetic_count(u))
        self.assertEqual(sum(counts), 128)
        self.assertEqual(sum(synthetic_count(u) for u in range(2000)), 51200)
        self.assertEqual(len(synthetic), 36)
        with self.assertRaises(RuntimeError):
            synthetic.add_episode(obs[0], np.zeros((4, 2)))

    def test_native_snapshot_reward_action_and_horizon(self):
        sim = TimedSimulator(26000001)
        sim.step(np.array([.7, -.2], np.float32))
        snapshot = sim.snapshot()
        other = TimedSimulator.restore(snapshot)
        action = np.array([.6, .1], np.float32)
        np.testing.assert_array_equal(sim.step(action)[0], other.step(action)[0])
        a = native(None, lambda p, obs: np.tile([.8, 0], (len(obs), 1)).astype(np.float32), [26000002, 26000003])
        b = native(None, lambda p, obs: np.tile([.8, 0], (len(obs), 1)).astype(np.float32), [26000002, 26000003])
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
        self.assertEqual(a['reward'].shape, (2, 50))
        np.testing.assert_allclose(a['action'], np.broadcast_to([.8, 0], (2, 50, 2)))


if __name__ == '__main__':
    unittest.main()
